import { PDFParse } from "pdf-parse";
import fs from "fs"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MistralAIEmbeddings } from '@langchain/mistralai'
import { Pinecone } from '@pinecone-database/pinecone'
import dotenv, { config } from "dotenv"
dotenv.config()

const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY
});

const index = pc.Index("cohort-2-rag")



// let dataBuffer = fs.readFileSync("./story.pdf")
// const parse = new PDFParse({
//     data: dataBuffer
// })

// const data = await parse.getText()

const embedding = new MistralAIEmbeddings({
    apiKey: process.env.MISTRAL_API_KEY,
    model: "mistral-embed"
})

// const spileter = new RecursiveCharacterTextSplitter({
//     chunkSize: 500,
//     chunkOverlap: 0
// })

// const chunks = await spileter.splitText(data.text)
// const docs = await Promise.all(chunks.map(async (chunk) => {
//     const embaddings = await embedding.embedQuery(chunk)
//     return {
//         text: chunk,
//         embaddings
//     }
// }))

// const result = await index.upsert({
//     records: docs.map((doc, index) => ({
//         id: `doc-${index}`,
//         values: doc.embaddings,
//         metadata: {
//             text: doc.text
//         }
//     }))
// })

const query = "how was the intership experience of arav"
const queryEmbedding = await embedding.embedQuery(query)

const result = await index.query({
    vector: queryEmbedding,
    topK: 3,
    includeMetadata: true
})
console.log(JSON.stringify(result))