// /* eslint-disable turbo/no-undeclared-env-vars */
import * as dotenv from "dotenv";
import {
  PGVectorStore,
  PostgresDocumentStore,
  PostgresIndexStore,
  SentenceSplitter,
  Settings,
  VectorStoreIndex,
  serviceContextFromDefaults,
  storageContextFromDefaults,
} from "llamaindex";
import path from "path";
import CustomTokenizer from "../customTokenizer";
import { DummyEmbedding } from "./DummyEmbedding";
import { getDocuments } from "./loader";
import { initSettings } from "./settings";
import {
  PGVECTOR_COLLECTION,
  PGVECTOR_SCHEMA,
  PGVECTOR_TABLE,
  checkRequiredEnvVars,
} from "./shared";
dotenv.config();

async function loadAndIndex() {
  // load objects from storage and convert them into LlamaIndex Document objects
  const documents = await getDocuments();

  // create postgres vector store
  const vectorStore = new PGVectorStore({
    connectionString: process.env.PG_CONNECTION_STRING,
    schemaName: PGVECTOR_SCHEMA,
    tableName: PGVECTOR_TABLE,
  });
  const indexStore = new PostgresIndexStore({
    connectionString: process.env.PG_CONNECTION_STRING,
    schemaName: PGVECTOR_SCHEMA,
  });

  const docStore = new PostgresDocumentStore({
    connectionString: process.env.PG_CONNECTION_STRING,
    schemaName: PGVECTOR_SCHEMA,
  });

  vectorStore.setCollection(PGVECTOR_COLLECTION);
  vectorStore.clearCollection();

  // Update the TOKENIZER_PATH
  const TOKENIZER_PATH = path.join(process.cwd(), "tokenizer_files");
  console.log("Tokenizer path:", TOKENIZER_PATH);
  const customTokenizer = new CustomTokenizer(TOKENIZER_PATH);
  await customTokenizer.initialize();

  // Create a custom text splitter
  const nodeParser = new SentenceSplitter({
    chunkSize: Settings.chunkSize,
    chunkOverlap: Settings.chunkOverlap,
    tokenizer: customTokenizer,
  });

  // Create a custom service context
  const serviceContext = serviceContextFromDefaults({
    chunkSize: Settings.chunkSize,
    chunkOverlap: Settings.chunkOverlap,
    nodeParser,
    embedModel: new DummyEmbedding(),
  });

  // create index from all the Documents
  console.log("Start creating embeddings...");
  const storageContext = await storageContextFromDefaults({
    docStore,
    indexStore,
    vectorStore,
  });
  // Use the custom service context when creating the index
  await VectorStoreIndex.fromDocuments(documents, {
    storageContext,
    serviceContext,
  });

  console.log(`Successfully created embeddings.`);
}

(async () => {
  checkRequiredEnvVars();
  initSettings();
  await loadAndIndex();
  console.log("Finished generating storage.");
  process.exit(0);
})();
