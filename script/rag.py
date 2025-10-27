import ollama
import pickle
import os
import time
import numpy as np
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
VECTOR_DB_FILE = 'vector_db.pkl'
CONFIG_FILE = 'config.json'

class VectorDB:
    def __init__(self, db_file: str = VECTOR_DB_FILE):
        self.db_file = db_file
        self.vectors = []
        self.chunks = []
        self.metadata = []
        self._load_db()
    
    def _load_db(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.vectors = np.array(data['vectors'])
                    self.chunks = data['chunks']
                    self.metadata = data['metadata']
                print(f'Загружена база данных: {len(self.chunks)} чанков, размер векторов: {self.vectors.shape}')
            except Exception as e:
                print(f'Ошибка загрузки БД: {e}, создаем новую')
                self.vectors = np.array([])
                self.chunks = []
                self.metadata = []
        else:
            print('База данных не найдена, создаем новую')
            self.vectors = np.array([])
            self.chunks = []
            self.metadata = []
    
    def save_db(self):
        try:
            data = {
                'vectors': self.vectors.tolist(),
                'chunks': self.chunks,
                'metadata': self.metadata,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            with open(self.db_file, 'wb') as f:
                pickle.dump(data, f)
            print(f'База данных сохранена: {len(self.chunks)} чанков')
        except Exception as e:
            print(f'Ошибка сохранения БД: {e}')
    
    def add_chunk(self, chunk: str, embedding: List[float], metadata: Dict = None):
        if metadata is None:
            metadata = {
                'added_at': datetime.now().isoformat(),
                'chunk_size': len(chunk),
                'source': 'cat-facts.txt'
            }
        
        embedding_array = np.array(embedding)
        
        if len(self.vectors) == 0:
            self.vectors = embedding_array.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, embedding_array])
        
        self.chunks.append(chunk)
        self.metadata.append(metadata)
    
    def batch_add_chunks(self, chunks_with_embeddings: List[Tuple[str, List[float], Dict]]):
        if not chunks_with_embeddings:
            return
        
        chunks, embeddings, metadatas = zip(*chunks_with_embeddings)
        
        new_vectors = np.array(embeddings)
        if len(self.vectors) == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        
        self.chunks.extend(chunks)
        self.metadata.extend(metadatas)
    
    def get_stats(self):
        if len(self.vectors) == 0:
            return "База данных пуста"
        
        total_chunks = len(self.chunks)
        vector_dim = self.vectors.shape[1] if len(self.vectors) > 0 else 0
        avg_chunk_size = np.mean([len(chunk) for chunk in self.chunks]) if self.chunks else 0
        
        return {
            'total_chunks': total_chunks,
            'vector_dimensions': vector_dim,
            'average_chunk_size': avg_chunk_size,
            'database_size_mb': os.path.getsize(self.db_file) / (1024 * 1024) if os.path.exists(self.db_file) else 0
        }

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def smart_chunking(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    if len(chunks) > 1 and overlap > 0:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
            else:
                prev_chunk_end = ' '.join(chunks[i-1].split()[-overlap:])
                new_chunk = prev_chunk_end + ' ' + chunks[i]
                overlapped_chunks.append(new_chunk)
        return overlapped_chunks
    
    return chunks

def load_and_process_dataset(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден!")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    chunks = smart_chunking(content)
    print(f'Загружено {len(chunks)} чанков из файла {file_path}')
    return chunks

def initialize_database(dataset: List[str], vector_db: VectorDB, batch_size: int = 10):
    if len(vector_db.chunks) > 0:
        print('База данных уже инициализирована, пропускаем создание')
        return
    
    print('Начинаем инициализацию базы данных...')
    
    batch_data = []
    for i, chunk in enumerate(dataset):
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            
            metadata = {
                'added_at': datetime.now().isoformat(),
                'chunk_size': len(chunk),
                'source': 'cat-facts.txt',
                'chunk_id': i
            }
            
            batch_data.append((chunk, embedding, metadata))
            
            if len(batch_data) >= batch_size:
                vector_db.batch_add_chunks(batch_data)
                batch_data = []
                print(f'Обработано {i+1}/{len(dataset)} чанков')
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f'Ошибка при обработке чанка {i}: {e}')
    
    if batch_data:
        vector_db.batch_add_chunks(batch_data)
    
    vector_db.save_db()
    print('База данных успешно инициализирована')

def retrieve_similar(vector_db: VectorDB, query: str, top_n: int = 3) -> List[Tuple[str, float]]:
    if len(vector_db.vectors) == 0:
        print("База данных пуста!")
        return []
    
    try:
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        query_array = np.array(query_embedding).reshape(1, -1)
        
        similarities = cosine_similarity(vector_db.vectors, query_array)
        
        top_indices = np.argsort(similarities.flatten())[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((vector_db.chunks[idx], float(similarities[idx])))
        
        return results
    
    except Exception as e:
        print(f'Ошибка при поиске: {e}')
        return []

def chat_with_context(vector_db: VectorDB, query: str, top_n: int = 3):
    print(f'🔍 Поиск релевантной информации для: "{query}"')
    
    retrieved_knowledge = retrieve_similar(vector_db, query, top_n)
    
    if not retrieved_knowledge:
        print('Не найдено релевантной информации')
        return
    
    print('\nНайденная информация:')
    for i, (chunk, similarity) in enumerate(retrieved_knowledge):
        print(f' {i+1}. (сходство: {similarity:.3f}) {chunk[:100]}...')
    
    context_chunks = [chunk for chunk, similarity in retrieved_knowledge]
    instruction_prompt = f'''Ты полезный ассистент. Используй только предоставленный контекст для ответа на вопрос.

Контекст:
{chr(10).join([f"- {chunk}" for chunk in context_chunks])}

Вопрос: {query}

Ответь на основе контекста. Если в контексте нет информации для ответа, скажи об этом.'''

    print('\nОтвет ассистента:')
    try:
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True,
        )
        
        response = ""
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            response += content
        
        return response
        
    except Exception as e:
        print(f'Ошибка при генерации ответа: {e}')
        return None

def main():
    print('Ассистент с оптимизированной векторной БД')
    print('=' * 50)
    
    vector_db = VectorDB()
    
    dataset = load_and_process_dataset('cat-facts.txt')
    if dataset:
        initialize_database(dataset, vector_db)
    
    stats = vector_db.get_stats()
    print('\nСтатистика базы данных:')
    for key, value in stats.items():
        print(f'   {key}: {value}')
    
    print('\nГотов к вопросам! (для выхода введите "quit")')
    print('-' * 50)
    
    while True:
        try:
            user_input = input('\nВаш вопрос: ').strip()
            
            if user_input.lower() in ['quit', 'exit', 'выход']:
                print('До свидания!')
                break
            
            if not user_input:
                continue
            
            chat_with_context(vector_db, user_input)
            
        except KeyboardInterrupt:
            print('\nДо свидания!')
            break
        except Exception as e:
            print(f'Произошла ошибка: {e}')

if __name__ == "__main__":
    main()
