import CustomTokenizer from "@/app/api/chat/customTokenizer";
import {
  Document,
  IngestionPipeline,
  SentenceSplitter,
  Settings,
  VectorStoreIndex,
} from "llamaindex";
import path from "path";

// Path to the directory containing your tokenizer files
const TOKENIZER_PATH = path.join(process.cwd(), "tokenizer_files");

export async function runPipeline(
  currentIndex: VectorStoreIndex,
  documents: Document[],
) {
  // Initialize the custom tokenizer
  const customTokenizer = new CustomTokenizer(TOKENIZER_PATH);
  await customTokenizer.initialize();

  // Use ingestion pipeline to process the documents into nodes and add them to the vector store
  const pipeline = new IngestionPipeline({
    transformations: [
      new SentenceSplitter({
        chunkSize: Settings.chunkSize,
        chunkOverlap: Settings.chunkOverlap,
        tokenizer: customTokenizer,
      }),
      Settings.embedModel,
    ],
  });
  const nodes = await pipeline.run({ documents });
  await currentIndex.insertNodes(nodes);
  currentIndex.storageContext.docStore.persist();
  console.log("Added nodes to the vector store.");
  return documents.map((document) => document.id_);
}
