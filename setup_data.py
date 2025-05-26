"""
Script to download and process MovieLens dataset
"""
from src.data_processor import MovieLensProcessor

def main():
    processor = MovieLensProcessor()
    
    # Download data
    print("Downloading MovieLens dataset...")
    processor.download_movielens()
      # Process all data
    print("Processing data...")
    processor.process_all()
    print("Done!")

if __name__ == "__main__":
    main()
