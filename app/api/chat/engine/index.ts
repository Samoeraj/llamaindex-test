import {
  ContextChatEngine,
  DeuceChatStrategy,
  LlamaDeuce,
  LLM,
  Ollama,
  serviceContextFromDefaults,
  SimpleDocumentStore,
  storageContextFromDefaults,
  VectorStoreIndex,
} from "llamaindex";
import { CHUNK_OVERLAP, CHUNK_SIZE, STORAGE_CACHE_DIR } from "./constants.mjs";

const ollamaLLM = new Ollama({model: "llama2", temperature: 0.75})

async function getDataSource(llm: LLM) {
  const serviceContext = serviceContextFromDefaults({
    llm: ollamaLLM,
    embedModel: ollamaLLM,
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
  });
  const storageContext = await storageContextFromDefaults({
    persistDir: `${STORAGE_CACHE_DIR}`,
  });

  const numberOfDocs = Object.keys(
    (storageContext.docStore as SimpleDocumentStore).toDict(),
  ).length;
  if (numberOfDocs === 0) {
    throw new Error(
      `StorageContext is empty - call 'npm run generate' to generate the storage first`,
    );
  }
  return await VectorStoreIndex.init({
    storageContext,
    serviceContext,
  });
}

export async function createChatEngine(llm: LLM) {
  const index = await getDataSource(llm);
  const retriever = index.asRetriever();
  retriever.similarityTopK = 3;

  return new ContextChatEngine({
    chatModel: llm,
    retriever,
  });
}
