const crypto = require('crypto');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const DriveDocumentIndex = require('../models/DriveDocumentIndex');
const PrdDocumentIndex = require('../models/PrdDocumentIndex');
const { readDriveFileContent } = require('./documentParser');
const { chunkPrdText } = require('./prdChunker');
const {
    isQdrantConfigured,
    buildPointId,
    upsertKnowledgePoints,
    deleteKnowledgeByFilter,
    searchKnowledge,
    getVerifiedVectorSize,
    getCollectionName,
    pingQdrant
} = require('./qdrantService');
const {hybridSearch} = require("./hybridsearch")

const CHUNK_SIZE = 1200;
const CHUNK_OVERLAP = 200;
const MAX_CHUNKS_PER_FILE = 200;
const DEFAULT_EMBEDDING_MODEL = 'gemini-embedding-001';
const FALLBACK_EMBEDDING_MODEL = 'gemini-embedding-2';
const DEFAULT_TOP_K = 5;
const DEFAULT_SCORE_THRESHOLD = 0.25;
const KEYWORD_BOOST = 0.15;

function getEmbeddingModelName() {
    return process.env.GEMINI_EMBEDDING_MODEL || DEFAULT_EMBEDDING_MODEL;
}

function getEmbeddingModel(modelName = getEmbeddingModelName()) {
    if (!process.env.GEMINI_API_KEY) {
        return null;
    }
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    return genAI.getGenerativeModel({ model: modelName });
}

/** Character-based chunking for Drive documents (existing strategy). */
function chunkText(text) {
    const chunks = [];
    if (!text) return chunks;

    let start = 0;
    while (start < text.length && chunks.length < MAX_CHUNKS_PER_FILE) {
        const end = Math.min(start + CHUNK_SIZE, text.length);
        const chunk = text.slice(start, end).trim();
        if (chunk) {
            chunks.push(chunk);
        }
        if (end >= text.length) break;
        start = end - CHUNK_OVERLAP;
    }

    return chunks;
}

async function embedText(text) {
    if (!process.env.GEMINI_API_KEY || !text?.trim()) {
        return [];
    }

    const modelsToTry = [getEmbeddingModelName(), FALLBACK_EMBEDDING_MODEL]
        .filter((model, index, arr) => arr.indexOf(model) === index);

    for (const modelName of modelsToTry) {
        try {
            const model = getEmbeddingModel(modelName);
            if (!model) return [];

            const result = await model.embedContent(text);
            const values = result.embedding?.values || [];
            if (values.length > 0) {
                return values;
            }
        } catch (error) {
            if (error.message?.includes('fetch failed') || error.code === 'ECONNREFUSED') {
                throw new Error(`Embedding request failed: embedding provider returned no response (${error.message}).`);
            }
            console.error(`[RAG] Embedding failed for model ${modelName}:`, error.message);
        }
    }

    return [];
}



function keywordScore(query, content) {
    const normalizedQuery = query.toLowerCase().trim();
    const normalizedContent = (content || '').toLowerCase();
    if (!normalizedQuery) return 0;
    if (normalizedContent.includes(normalizedQuery)) return 0.75;

    const terms = normalizedQuery.split(/\s+/).filter(Boolean);
    if (!terms.length) return 0;

    const matched = terms.filter((term) => normalizedContent.includes(term)).length;
    return matched / terms.length;
}

function hashContent(text) {
    return crypto.createHash('sha256').update(text || '').digest('hex');
}

function formatSourceLabel(payload) {
    const sourceType = payload.sourceType || 'unknown';
    if (sourceType === 'prd') {
        const section = payload.section || 'Document';
        return `PRD → ${section}`;
    }
    if (sourceType === 'drive') {
        return payload.fileName || 'Google Drive document';
    }
    if (sourceType === 'github') {
        const meta = payload.metadata || {};
        const githubPath = meta.path || payload.fileName || payload.section;
        return githubPath ? `GitHub → ${githubPath}` : 'GitHub';
    }
    return payload.fileName || 'Project knowledge';
}

async function buildQdrantPoints(userId, projectId, sourceType, sourceDocumentId, chunks, extraPayload = {}) {
    const points = [];
    let vectorSize = null;

    for (const chunk of chunks) {
        const text = chunk.text || chunk.content || chunk;
        const section = chunk.section || extraPayload.section || '';
        const chunkIndex = chunk.chunkIndex ?? points.length;

        const embedding = await embedText(typeof text === 'string' ? text : String(text));
        const sparseEmbedding = {
            text,
            model:"qdrant/bm25"
        }
        if (!embedding.length) {
            console.warn(`[RAG] Skipping chunk ${chunkIndex} — empty embedding`);
            continue;
        }

        if (!vectorSize) vectorSize = embedding.length;

        const payload = {
            userId: String(userId),
            projectId: String(projectId),
            sourceType,
            sourceDocumentId: String(sourceDocumentId),
            section,
            chunkIndex,
            text: typeof text === 'string' ? text : String(text),
            fileName: extraPayload.fileName || '',
            webViewLink: extraPayload.webViewLink || '',
            metadata: extraPayload.metadata || {}
        };

        points.push({
            id: buildPointId(userId, projectId, sourceType, sourceDocumentId, chunkIndex),
            vector: {
                dense:embedding,
                sparse: sparseEmbedding,
            },
            payload
        });
    }

    return { points, vectorSize };
}

/**
 * Index complete PRD text into Qdrant (semantic chunking).
 */
async function indexPrdDocument(userId, projectId, prdText, options = {}) {
    const { filename = 'prd', sourceDocumentId = `prd-${projectId}` } = options;

    console.log(`[RAG] Starting PRD indexing for project ${projectId}`);

    if (!isQdrantConfigured()) {
        console.warn(`[RAG] Qdrant is not configured (QDRANT_URL missing).`);
        return { indexed: false, reason: 'Qdrant is not configured (QDRANT_URL missing).' };
    }

    console.log(`[RAG] Embedding provider: Gemini (${getEmbeddingModelName()})`);
    console.log(`[RAG] Qdrant endpoint: ${process.env.QDRANT_URL}`);
    console.log(`[RAG] Collection: ${getCollectionName()}`);

    const isReachable = await pingQdrant();
    if (!isReachable) {
        console.warn(`[RAG] Qdrant unavailable at ${process.env.QDRANT_URL}`);
        return { indexed: false, reason: 'Qdrant is configured but unreachable.' };
    }

    if (!prdText || prdText.trim().length < 50) {
        return { indexed: false, reason: 'PRD text too short to index.' };
    }

    try {
        const contentHash = hashContent(prdText);
        const existing = await PrdDocumentIndex.findOne({ userId, projectId });

        if (existing && existing.contentHash === contentHash && existing.status === 'indexed') {
            console.log(`[RAG] PRD already indexed and unchanged for project ${projectId}`);
            return {
                indexed: false,
                reason: 'PRD already indexed and unchanged.',
                chunkCount: existing.chunkCount,
                sourceDocumentId: existing.sourceDocumentId
            };
        }

        const semanticChunks = chunkPrdText(prdText);
        if (!semanticChunks.length) {
            return { indexed: false, reason: 'No semantic chunks produced from PRD.' };
        }
        console.log(`[RAG] Chunk count: ${semanticChunks.length}`);

        await deleteKnowledgeByFilter(userId, projectId, { sourceType: 'prd' });

        console.log(`[RAG] Embedding request started`);
        const { points, vectorSize } = await buildQdrantPoints(
            userId,
            projectId,
            'prd',
            sourceDocumentId,
            semanticChunks,
            { fileName: filename, metadata: { contentHash } }
        );

        if (!points.length || !vectorSize) {
            return { indexed: false, reason: 'Failed to generate embeddings for PRD chunks.' };
        }

        console.log(`[RAG] Qdrant upsert started`);
        await upsertKnowledgePoints(points, vectorSize);
        

        const indexRecord = await PrdDocumentIndex.findOneAndUpdate(
            { userId, projectId },
            {
                sourceDocumentId,
                filename,
                contentHash,
                chunkCount: points.length,
                vectorDimension: vectorSize,
                status: 'indexed',
                indexedAt: new Date(),
                error: null,
                version: (existing?.version || 0) + 1
            },
            { upsert: true, new: true }
        );

        console.log(`[RAG] PRD indexing complete`);
        console.log(`[RAG] Chunks: ${points.length}`);
        console.log(`[RAG] Embeddings: ${vectorSize} dimensions`);
        console.log(`[RAG] Qdrant upsert: success`);

        return {
            indexed: true,
            chunkCount: points.length,
            vectorDimension: vectorSize,
            sourceDocumentId,
            collection: getCollectionName(),
            indexRecord
        };
    } catch (error) {
        console.error('[RAG] indexPrdDocument failed:', error.message);
        await PrdDocumentIndex.findOneAndUpdate(
            { userId, projectId },
            { status: 'failed', error: error.message, indexedAt: new Date() },
            { upsert: true }
        );
        return { indexed: false, reason: 'PRD indexing failed.', error: error.message };
    }
}

/**
 * Index Google Drive file into Qdrant (character chunking — existing behavior).
 */
async function indexDriveFile(userId, projectId, fileId) {
    if (!isQdrantConfigured()) {
        return { indexed: false, reason: 'Qdrant is not configured (QDRANT_URL missing).' };
    }

    try {
        const content = await readDriveFileContent(userId, fileId);
        if (!content.text || content.text.length < 20) {
            return { indexed: false, reason: 'No extractable text content.' };
        }

        const contentHash = hashContent(content.text);
        const existing = await DriveDocumentIndex.findOne({ userId, driveFileId: fileId });
        if (existing && existing.contentHash === contentHash) {
            return { indexed: false, reason: 'File already indexed and unchanged.', file: existing };
        }

        const textChunks = chunkText(content.text);
        const semanticChunks = textChunks.map((text, i) => ({
            text,
            section: content.metadata?.name || 'Drive Document',
            chunkIndex: i
        }));

        await deleteKnowledgeByFilter(userId, projectId, {
            sourceType: 'drive',
            sourceDocumentId: fileId
        });

        const { points, vectorSize } = await buildQdrantPoints(
            userId,
            projectId,
            'drive',
            fileId,
            semanticChunks,
            {
                fileName: content.metadata.name,
                webViewLink: content.metadata.webViewLink || '',
                metadata: { contentHash, mimeType: content.metadata.mimeType }
            }
        );

        if (!points.length || !vectorSize) {
            return { indexed: false, reason: 'Failed to generate embeddings for Drive file.' };
        }

        await upsertKnowledgePoints(points, vectorSize);

        const indexRecord = await DriveDocumentIndex.findOneAndUpdate(
            { userId, driveFileId: fileId },
            {
                projectId: projectId || null,
                name: content.metadata.name,
                mimeType: content.metadata.mimeType,
                webViewLink: content.metadata.webViewLink || '',
                modifiedTime: content.metadata.modifiedTime ? new Date(content.metadata.modifiedTime) : null,
                size: content.metadata.size || 0,
                parents: content.metadata.parents || [],
                indexedAt: new Date(),
                chunkCount: points.length,
                contentHash,
                vectorStore: 'qdrant'
            },
            { upsert: true, new: true }
        );

        return { indexed: true, chunkCount: points.length, file: indexRecord };
    } catch (error) {
        console.error('[RAG] indexDriveFile failed:', error.message);
        return { indexed: false, reason: 'Indexing failed.', error: error.message };
    }
}

/**
 * Hybrid search across PRD + Drive knowledge in Qdrant.
 */
async function searchProjectKnowledge(userId, query, options = {}) {
    const limit = options.limit || DEFAULT_TOP_K;
    const scoreThreshold = options.scoreThreshold ?? DEFAULT_SCORE_THRESHOLD;

    if (!isQdrantConfigured()) {
        return [];
    }

    if (!options.projectId) {
        console.warn('[RAG] searchProjectKnowledge called without projectId — returning empty');
        return [];
    }

    const queryEmbedding = await embedText(query);
    if (!queryEmbedding.length) {
        return [];
    }

    const filterExtra = {};
    if (options.sourceType) {
        filterExtra.sourceType = options.sourceType;
    }

    const vectorResults = await searchKnowledge(
        userId,
        options.projectId,
        queryEmbedding,
        query,
        { limit: limit * 2, scoreThreshold, filterExtra }
    );

    const scored = vectorResults.map((result) => {
        const payload = result.payload || {};
        const text = payload.text || '';
        const keyword = keywordScore(query, text);
        const combinedScore = result.score + (keyword > 0.2 ? KEYWORD_BOOST * keyword : 0);

        return {
            score: combinedScore,
            vectorScore: result.score,
            keywordScore: keyword,
            sourceType: payload.sourceType || 'unknown',
            sourceDocumentId: payload.sourceDocumentId || '',
            section: payload.section || '',
            fileId: payload.sourceType === 'drive' ? payload.sourceDocumentId : undefined,
            fileName: payload.fileName || formatSourceLabel(payload),
            webViewLink: payload.webViewLink || '',
            content: text,
            sourceLabel: formatSourceLabel(payload),
            path: payload.metadata?.path || undefined,
            repository: payload.metadata?.repo
                ? `${payload.metadata.owner}/${payload.metadata.repo}`
                : undefined,
            commitSha: payload.metadata?.commitSha || undefined
        };
    });

    return scored
        .filter((item) => item.score >= scoreThreshold)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
}

async function getKnowledgeContext(userId, projectId, query) {
    try {
        const results = await searchProjectKnowledge(userId, query, { projectId, limit: 5 });
        if (!results.length) {
            return { context: '', sources: [], ragAvailable: isQdrantConfigured() };
        }

        const context = results
            .map((result, index) =>
                `[Source ${index + 1}: ${result.sourceLabel}]\n${result.content}`
            )
            .join('\n\n');

        const sources = results.map((result) => ({
            title: result.sourceLabel,
            url: result.webViewLink || undefined,
            fileId: result.fileId,
            sourceType: result.sourceType,
            section: result.section,
            path: result.path,
            commitSha: result.commitSha
        }));

        return { context, sources, ragAvailable: true };
    } catch (error) {
        console.error('[RAG] getKnowledgeContext failed:', error.message);
        return { context: '', sources: [], ragAvailable: false };
    }
}

async function deletePrdKnowledge(userId, projectId) {
    await deleteKnowledgeByFilter(userId, projectId, { sourceType: 'prd' });
    await PrdDocumentIndex.findOneAndUpdate(
        { userId, projectId },
        { status: 'deleted', chunkCount: 0 }
    );
}

module.exports = {
    indexPrdDocument,
    indexDriveFile,
    searchProjectKnowledge,
    getKnowledgeContext,
    deletePrdKnowledge,
    chunkText,
    chunkPrdText,
    embedText,
    hashContent,
    buildQdrantPoints,
    getEmbeddingModelName,
    getVerifiedVectorSize,
    isQdrantConfigured,
    getCollectionName,
    formatSourceLabel,
    DEFAULT_TOP_K,
    DEFAULT_SCORE_THRESHOLD
};
