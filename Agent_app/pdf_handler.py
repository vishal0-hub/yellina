import fitz  # PyMuPDF
import os
import uuid
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re


class PdfHandler:
    def __init__(self,pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.results = []
        self.full_text = ""     # full book text
        self.pdf_type = ""      # book or schedule
    
    def extract_images(self,output_folder):
        #  create the directory if not exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_count = 0
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                bbox = page.get_image_bbox(img) 
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_uuid = str(uuid.uuid4())
                image_filename = f"image_{image_uuid}.{image_ext}"
                image_filepath = f"{output_folder}/{image_filename}"
                # print("image_filepath--->>>",image_filepath)               
                with open(image_filepath, "wb") as img_file:
                    img_file.write(image_bytes)
                
                image_count += 1

                #  get nerby  text
                expand=50
                rect = fitz.Rect(bbox.x0, bbox.y1, bbox.x1, bbox.y1 + expand)
                text = page.get_textbox(rect)                
                #  apend data to the results
                self.results.append({
                    "image_path": image_filepath,
                    "page": page_num + 1,
                    "text": text.strip()
                })
            
        return self.results
    
    def extract_full_text(self):
        """Extract the full book text (all pages concatenated)."""
        all_text = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text("text")
            if text.strip():
                cleaned_text = text.replace("\u200b", "").strip().lower()

                all_text.append(cleaned_text)
        self.full_text = "\n".join(all_text)

        #  detect the pdf type
        self.detect_pdf_type()

        return self.full_text
    
    #  detect the pdf type
    def detect_pdf_type(self,):
        if re.search(r'(Sala|Room|sala|room) [A-Za-z]', self.full_text):
            self.pdf_type = "schedule"
        else:
            self.pdf_type = "book"

        print('pdf_type-->>', self.pdf_type)    

        return self.pdf_type
    # old chunk code
    # Split book text into chunks
    # def chunk_text(self, text, chunk_size=500, overlap=50):
        # """
        # Split text into chunks with word overlap.
        
        # :param text: Input string
        # :param chunk_size: Max words per chunk
        # :param overlap: Number of words to overlap between chunks
        # """
        # words = text.split()
        # start = 0
        # while start < len(words):
        #     end = start + chunk_size
        #     chunk = words[start:end]
        #     yield " ".join(chunk)
        #     start += chunk_size - overlap 
    
    # new chunk code
    # def chunk_text(self, text, chunk_size=500, overlap=50):
    #     """
    #     Split text into chunks with word overlap.
        
    #     :param text: Input string
    #     :param chunk_size: Max words per chunk
    #     :param overlap: Number of words to overlap between chunks
    #     """
    #     # Split text into sentences based on punctuation
    #     sentences = re.split(r'(?<=[.!?])\s+', text)  # works for English & Italian
    #     chunks = []
    #     current_chunk = []
    #     current_len = 0

    #     for sentence in sentences:
    #         words = sentence.split()
    #         if current_len + len(words) > chunk_size:
    #             # finalize current chunk
    #             chunks.append(" ".join(current_chunk))
    #             # start new chunk with overlap
    #             overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
    #             current_chunk = overlap_words + words
    #             current_len = len(current_chunk)
    #         else:
    #             current_chunk.extend(words)
    #             current_len += len(words)

    #     # add the last chunk
    #     if current_chunk:
    #         chunks.append(" ".join(current_chunk))
        
    #     print('chunk-->>', chunks)

    #     return chunks
    
    #  chunk  test
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into chunks with word overlap using RecursiveCharacterTextSplitter."""

        if self.pdf_type == "schedule":
            # For schedule, first split into sections based on "Sala" or "Room"
            sections = self.split_schedule_into_sections(text)
            # all_chunks = []
            # for section in sections:
            #     chunks = self._split_text_with_splitter(section, chunk_size, overlap)
            #     all_chunks.extend(chunks)
            print('length--->>>', len(sections))
            return sections
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n\n","\n\n", "\n"]
            )
            chunks = text_splitter.split_text(text)
            return chunks
    
    #  split scheduele text into sections
    def split_schedule_into_sections(self, text):
        # Find all block start positions
        # matches = list(re.finditer(r'(Sala|Room) [A-Z]', text))
        matches = list(re.finditer(r'(Sala|Room|sala|room) [A-Za-z]', text))

        blocks = []
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            block_text = text[start:end].strip()
            blocks.append(block_text)
        
        # print('blocks-->>', blocks)
        return blocks

    def extract_metadata(self, text):
        room_match = re.search(r"(Sala|Room)\s+[A-Za-z]", text, re.IGNORECASE)
        date_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)
        time_match = re.search(r"\b\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?\b", text)

        # print('room_match-->>', room_match)
        # print('date_match-->>', date_match)
        # print('time_match-->>', time_match)

        return {
            "room": room_match.group(0).lower() if room_match else None,
            "date": date_match.group(0) if date_match else None,
            "time": time_match.group(0) if time_match else None,
        }
        

        



if __name__ == "__main__":
    _obj = PdfHandler("..//media//uploads//schedule_pJOzERW.pdf")
    chunks=_obj.chunk_text(text=_obj.extract_full_text(), chunk_size=100, overlap=20)
    
    # extract meta data dtal
    # print('chunks--->>', chunks)
    for chunk in chunks:
        print('\n\n', _obj.extract_metadata(chunk))
    # print(_obj.extract_metadata(_obj.full_text))



