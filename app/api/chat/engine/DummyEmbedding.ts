import { BaseEmbedding, EmbeddingInfo } from "llamaindex";

export class DummyEmbedding extends BaseEmbedding {
  dimensions: number;

  constructor(dimensions: number = 1536) {
    super();
    this.dimensions = dimensions;
    this.embedInfo = {
      dimensions: this.dimensions,
    };
  }

  async getTextEmbedding(text: string): Promise<number[]> {
    return this.generateDummyEmbedding();
  }

  getTextEmbeddings = async (texts: string[]): Promise<number[][]> => {
    return texts.map(() => this.generateDummyEmbedding());
  };

  private generateDummyEmbedding(): number[] {
    return Array.from({ length: this.dimensions }, () => Math.random());
  }
}