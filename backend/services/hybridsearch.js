const {
  embedText
} = require("./ragService");

const {getClient, getCollectionName} = require("./qdrantService")

async function hybridSearch({
  queryText,UserID,ProjectID,Limit = 10
}){
  if(!query || !queryText.trim()){
    return [];
  }
}
 
const qdrant = getClient();
const collectionname = getCollectionName()

const denseVector = embedText(queryText)

const queryTextVector = {
  text:queryText,
  model:"qdrant/bm25",
}

const filter = {

  must :[
    {
      key:"userId",
      match:{
        value: String(userId)
      },
    },
    {
      key:"projectId",
      match:{
        value: String(projectId)
      },
    }
  ]

}

const result = qdrant.query(collectionname, {
  prefetch:[
    {
      query:denseVector,
      using: "dense",
      limit : Math.max(limit *2,20)
    },
    {
      query:sparseVector,
      using: "sparse",
      limit : Math.max(limit *2,20)
    }
  ]
})

module.exports = {
  hybridSearch
}