from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
from sentence_transformers import SentenceTransformer
import re
import spacy

nlp = spacy.load("it_core_news_sm")

#  old embeder
# embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#new multilingual embeder
# embedder  = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings to unit length
)

class Embedder:
    def __init__(self, index_path=None, ftype="pdf"):
        self.index = None
        self.docs = []
        self.index_path = index_path
        self.ftype=ftype
    
    def preprocess_italian(self, text):
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop])
        
    # convert  text to document
    def text_to_docs(self, data):
        docs = []
        for item in data:
            if self.ftype=="csv":
                page_content=item["text"]
            else:
                page_content=self.preprocess_italian(item["text"])

            docs.append(Document(
                
                page_content=page_content,
                metadata={
                    "type": item["type"],
                    "book": item["book"],
                "page": item.get("page"),
                "image_path": item.get("image_path"),
                "room": item.get("room"),
            }
        ))
            
        self.docs = docs
        print(f"Converted {len(docs)} items to Document format")
            
        return True
    
    def create_faiss_index(self):
        try:
            if not self.index_path:
                return "index_path not provided"
            if not self.docs:
                return "No documents to index"
            
            
            self.index = FAISS.from_documents(self.docs, embedder)
            self.index.save_local(self.index_path)
            print(f"Index created and saved to {self.index_path}")

            return True
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            return False
        
    # load existing index
    def load_faiss_index(self):
        try:
            if not self.index_path:
                return "index_path not provided"
            
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"⚠️ No index found at {self.index_path}")
            
            self.index = FAISS.load_local(self.index_path, embedder, allow_dangerous_deserialization=True)
            print("Index loaded successfully")
            return True
        
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            self.index = None
            return False
    
    # Incrementally add docs
    def add_documents(self, new_data):
        try:
            if not self.index:
                print("⚠️ No FAISS index loaded/created")
                return False

            new_docs = self.text_to_docs(new_data)
            if new_docs:
                self.index.add_documents( self.docs)
                self.index.save_local(self.index_path)
                print(f"✅ Added {len(self.docs)} new documents and updated index")
                return True
            
            return False
        except Exception as e:
            print(f"Error adding documents to FAISS index: {e}")
            return False

    # query the index
    def query_index(self, query, k=4):
        try:
            print(f"Querying index for: {query}")
            print(f"self.index: {self.index}")
            if not self.index:
                print("⚠️ No FAISS index loaded")
                return False

            results=[]
            
            #  detect is there any room in  query
            room_match = re.search(r"\b(Sala|Room)\b\s+[A-Za-z]+", query, re.IGNORECASE)

            print(f"room_match: {room_match}")
            if room_match:
                room_name = room_match.group(0).lower()
                print(f"Detected room name: {room_name}")
                # Loop through all documents in the index
                for doc in self.index.docstore._dict.values():
                    if doc.metadata.get("room") == room_name:
                        results.append(doc)

                # results.extend(self.index.similarity_search(
                #                 query=query,   # dummy query, not really used
                #                 k=50,
                #                 filter={"room": room_name}   # ✅ direct metadata filter
                #             ))
                print('\n\nresults with room filter-->>', results)
            
            if not results or  not room_match   :
                results.extend(self.index.similarity_search(query, k=k))
                print('results-->>', results)

            return results
        except Exception as e:
            print(f"Error querying FAISS index: {e}")
            return []
        
    def delete_book(self, book_name):
        # keep only docs not matching book_name
        remaining_docs = [doc for doc in self.index.docstore._dict.values() if doc.metadata["book"] != book_name]

        # rebuild FAISS index
        self.index = FAISS.from_documents(remaining_docs, embedder)
        self.index.save_local(self.index_path)
        return f"Book '{book_name}' removed from index"



if __name__ == "__main__":
    embedder_obj = Embedder(index_path="77_faiss_index")
    # D:\Aashish\AI-Agents\59_faiss_index
    embedder_obj.load_faiss_index()
    # query="""c'è qualche sessione nella sala i?"""
    
    # query=" Che cos'è la capsuloressi?"   
    query="Che cos'è la facoemulsificazione?"

    # results = embedder_obj.query_index(query, k=4)
    # for res in results:
    #     print(res.page_content, res.metadata)

    for doc in embedder_obj.index.docstore._dict.values():
        print(doc.page_content)

