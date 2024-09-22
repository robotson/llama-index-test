import { AutoTokenizer, env } from '@xenova/transformers';
import path from 'path';

class CustomTokenizer {
  private tokenizer: any;
  private tokenizerPath: string;

  constructor(tokenizerPath: string) {
    this.tokenizerPath = tokenizerPath;
  }

  async initialize() {
    console.log('Initializing tokenizer from:', this.tokenizerPath);
    env.localModelPath = this.tokenizerPath;
    this.tokenizer = await AutoTokenizer.from_pretrained("", { 
      local_files_only: true,
    });
  }

  encode(text: string): Uint32Array {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialized. Call initialize() first.');
    }
    const encoded = this.tokenizer.encode(text);
    // console.log("in encode", encoded);
    return new Uint32Array(encoded);
  }

  decode(tokens: Uint32Array): string {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialized. Call initialize() first.');
    }
    return this.tokenizer.decode(Array.from(tokens));
  }
}

export default CustomTokenizer;
