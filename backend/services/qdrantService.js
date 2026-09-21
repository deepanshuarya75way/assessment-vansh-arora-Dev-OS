const crypto = require('crypto');
const { QdrantClient } = require('@qdrant/js-client-rest');

const DEFAULT_COLLECTION = 'devos_knowledge';
const DISTANCE = 'Cosine';

let client = null;
let collectionReady = false;
let verifiedVectorSize = null;

function getCollectionName() {
    return process.env.QDRANT_COLLECTION_NAME || DEFAULT_COLLECTION;
}

function isQdrantConfigured() {
    return Boolean(process.env.QDRANT_URL);
}

async function pingQdrant() {
    if (!isQdrantConfigured()) return false;
    try {
        const qdrant = getClient();
        await qdrant.getCollections();
        return true;
    } catch (err) {
        return false;
    }
}

function getClient() {
    if (!isQdrantConfigured()) {
        throw new Error('Qdrant is not configured. Set QDRANT_URL in environment.');
    }
    if (!client) {
        const options = { url: process.env.QDRANT_URL };
        if (process.env.QDRANT_API_KEY) {
            options.apiKey = process.env.QDRANT_API_KEY;
        }
        client = new QdrantClient(options);
    }
    return client;
}

function buildPointId(userId, projectId, sourceType, sourceDocumentId, chunkIndex) {
    const raw = `${userId}:${projectId}:${sourceType}:${sourceDocumentId}:${chunkIndex}`;
    const hash = crypto.createHash('sha256').update(raw).digest('hex');
    return hash.slice(0, 32);
}

const PAYLOAD_INDEX_FIELDS = ['userId', 'projectId', 'sourceType', 'sourceDocumentId'];

async function ensurePayloadIndexes(qdrant, collectionName) {
    for (const field_name of PAYLOAD_INDEX_FIELDS) {
        try {
            await qdrant.createPayloadIndex(collectionName, {
                wait: true,
                field_name,
                field_schema: 'keyword'
            });
        } catch (err) {
            // Index may already exist — safe to ignore duplicate errors
            if (!err.message?.includes('already exists')) {
                console.warn(`[Qdrant] Payload index warning for ${field_name}:`, err.message);
            }
        }
    }
}

async function ensureCollection(vectorSize) {
    if (!vectorSize || vectorSize < 1) {
        throw new Error('Invalid vector dimension for Qdrant collection');
    }

    const qdrant = getClient();
    const collectionName = getCollectionName();

    if (collectionReady && verifiedVectorSize === vectorSize) {
        return collectionName;
    }

    try {
        const collections = await qdrant.getCollections();
        const exists = collections.collections?.some((c) => c.name === collectionName);

        if (!exists) {
            await qdrant.createCollection(collectionName, {
                vectors: {
                    dense:{
                        size: vectorSize,
                        distance : "COSINE"
                    }
                },
                sparse_vector:{
                    sparse:{
                        mdoifier : "idf"
                    }
                }
            });
            console.log(`[Qdrant] Created collection "${collectionName}" (dim=${vectorSize})`);
        } else {
            const info = await qdrant.getCollection(collectionName);
            const existingSize = info.config?.params?.vectors?.size;
            if (existingSize && existingSize !== vectorSize) {
                throw new Error(
                    `Qdrant collection "${collectionName}" vector size mismatch: expected ${vectorSize}, found ${existingSize}`
                );
            }
        }
    } catch (err) {
        if (err.message?.includes('fetch failed') || err.code === 'ECONNREFUSED') {
            throw new Error(`Qdrant connection failed: unable to reach configured Qdrant endpoint (${process.env.QDRANT_URL}).`);
        }
        throw err;
    }

    await ensurePayloadIndexes(qdrant, collectionName);

    verifiedVectorSize = vectorSize;
    collectionReady = true;
    return collectionName;
}

function buildFilter(userId, projectId, extra = {}) {
    const must = [
        { key: 'userId', match: { value: String(userId) } },
        { key: 'projectId', match: { value: String(projectId) } }
    ];

    if (extra.sourceType) {
        must.push({ key: 'sourceType', match: { value: extra.sourceType } });
    }
    if (extra.sourceDocumentId) {
        must.push({ key: 'sourceDocumentId', match: { value: String(extra.sourceDocumentId) } });
    }

    return { must };
}

async function upsertKnowledgePoints(points, vectorSize) {
    if (!points.length) return { upserted: 0 };

    const collectionName = await ensureCollection(vectorSize);
    const qdrant = getClient();

    try {
        await qdrant.upsert(collectionName, {
            wait: true,
            points
        });
    } catch (err) {
        if (err.message?.includes('fetch failed') || err.code === 'ECONNREFUSED') {
            throw new Error(`Qdrant connection failed: unable to reach configured Qdrant endpoint (${process.env.QDRANT_URL}).`);
        }
        throw err;
    }

    return { upserted: points.length, collection: collectionName };
}

async function deleteKnowledgeByFilter(userId, projectId, filterExtra = {}) {
    if (!isQdrantConfigured()) return { deleted: false, reason: 'Qdrant not configured' };

    const collectionName = getCollectionName();
    const qdrant = getClient();
    const filter = buildFilter(userId, projectId, filterExtra);

    // Indexes required for filter-based delete
    if (verifiedVectorSize) {
        await ensureCollection(verifiedVectorSize);
    } else {
        await ensurePayloadIndexes(qdrant, collectionName);
    }

    try {
        await qdrant.delete(collectionName, {
            wait: true,
            filter
        });
        return { deleted: true };
    } catch (err) {
        if (err.message?.includes('Not found') || err.status === 404) {
            return { deleted: false, reason: 'Collection not found' };
        }
        throw err;
    }
}

async function searchKnowledge(userId, projectId, queryVector, options = {}) {
    if (!isQdrantConfigured()) {
        return [];
    }

    const limit = options.limit || 5;
    const scoreThreshold = options.scoreThreshold ?? 0.25;
    const filterExtra = options.filterExtra || {};

    const collectionName = getCollectionName();
    const qdrant = getClient();
    const filter = buildFilter(userId, projectId, filterExtra);

    // Ensure payload indexes exist (required for filtered queries on Qdrant Cloud)
    if (queryVector?.length) {
        await ensureCollection(queryVector.length);
    }

    try {
        const response = await qdrant.query(collectionName, {
            prefetch : [
                {
                    query: queryVector,
                    using: "dense",
                    filter,
                    limit: limit*2,
                },
                {
                    query:{
                        text: queryText,
                        model:"qdrant/bm25"
                    },
                    using:"sparse",
                    filter,
                    limit: limit*2
                },
            ],
            query:{
                rrf:{},
            },
            limit,
            with_payload:true
        });

        const points = response?.points || [];

        return points.map((r) => ({
            id: r.id,
            score: r.score,
            payload: r.payload || {}
        }));
    } catch (err) {
        if (err.message?.includes('Not found') || err.status === 404) {
            return [];
        }
        throw err;
    }
}

function getVerifiedVectorSize() {
    return verifiedVectorSize;
}

module.exports = {
    DEFAULT_COLLECTION,
    isQdrantConfigured,
    getCollectionName,
    getClient,
    buildPointId,
    ensureCollection,
    buildFilter,
    upsertKnowledgePoints,
    deleteKnowledgeByFilter,
    searchKnowledge,
    getVerifiedVectorSize,
    pingQdrant
};
