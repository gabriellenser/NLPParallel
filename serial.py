import csv
import re
import math
import time
import hashlib
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer
import nltk
from scipy.sparse import csr_matrix
import numpy as np
import psutil
from monitorr import Monitor
from pathlib import Path


try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class processing:
    
    def __init__(self, min_df=10, max_df=0.9, max_features=50000, porcentagem=100, coluna='text'):
        self.lemmatizer = WordNetLemmatizer()
        
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.porcentagem = porcentagem
        self.coluna = coluna
        
        self.vocabulario = {}
        self.idf = {}
        
        self.tempo_construcao_vocab = 0.0
        self.tempo_tokenizacao = 0.0
        self.tempo_lematizacao = 0.0
        self.tempo_vetorizacao = 0.0
        self.tempo_execucao_total = 0.0  
        self.num_documentos = 0
        self.total_tokens = 0
        
        self.processo = psutil.Process()
        self.memoria_inicial = 0
        self.memoria_pico = 0
        self.memoria_final = 0
    
    def atualizar_pico(self):
        memoria_atual = self.processo.memory_info().rss
        if memoria_atual > self.memoria_pico:
            self.memoria_pico = memoria_atual
    
    def contar_linhas(self, caminho_arquivo):
        with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
            leitor = csv.DictReader(arquivo)
            return sum(1 for linha in leitor if self.coluna in linha and linha[self.coluna])
    
    def ler_csv(self, caminho_arquivo, max_lines=None):
        with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
            leitor = csv.DictReader(arquivo)
            
            linhas_lidas = 0
            for linha in leitor:
                if max_lines and linhas_lidas >= max_lines:
                    break
                if self.coluna in linha and linha[self.coluna]:
                    yield linha[self.coluna]
                    linhas_lidas += 1
    
    def linhas_para_processar(self, caminho_arquivo):
        total_linhas = self.contar_linhas(caminho_arquivo)
        return int((self.porcentagem / 100) * total_linhas)
    
    def tokenizar(self, texto):
        if not texto:
            return []
        texto_lower = texto.lower()
        return re.findall(r'\b[a-z0-9]+\b', texto_lower)
    
    def lematizar(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def carregar_dados(self, caminho_arquivo):
        max_docs = self.linhas_para_processar(caminho_arquivo)
        self.memoria_inicial = self.processo.memory_info().rss
        self.memoria_pico = self.memoria_inicial
        
        textos = list(self.ler_csv(caminho_arquivo, max_lines=max_docs))
        
        return textos
    
    def processar_documentos(self, textos):
        
        inicio_tokenizacao = time.perf_counter()
        
        todos_tokens = []
        total_tokens = 0
        
        for texto in textos:
            tokens = self.tokenizar(texto)
            todos_tokens.append(tokens)
            total_tokens += len(tokens)
        
        fim_tokenizacao = time.perf_counter()
        self.tempo_tokenizacao = fim_tokenizacao - inicio_tokenizacao
        self.total_tokens = total_tokens
        
        inicio_lematizacao = time.perf_counter()
        
        documentos_processados = []
        for tokens in todos_tokens:
            if tokens:
                tokens_lematizados = self.lematizar(tokens)
                documentos_processados.append(tokens_lematizados)
            else:
                documentos_processados.append([])
        
        fim_lematizacao = time.perf_counter()
        self.tempo_lematizacao = fim_lematizacao - inicio_lematizacao
        self.num_documentos = len(documentos_processados)
        
        return documentos_processados
    
    def construir_vocabulario(self, documentos_processados):
        inicio = time.perf_counter()
        
        doc_freq = defaultdict(int)
        num_docs = 0
        
        for tokens_lematizados in documentos_processados:
            if tokens_lematizados:
                num_docs += 1
                for palavra in set(tokens_lematizados):
                    doc_freq[palavra] += 1
        
        max_doc_freq = int(self.max_df * num_docs)
        vocab_filtrado = {
            palavra: freq 
            for palavra, freq in doc_freq.items() 
            if self.min_df <= freq <= max_doc_freq
        }
        
        if len(vocab_filtrado) > self.max_features:
            palavras_top = sorted(
                vocab_filtrado.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.max_features]
            vocab_filtrado = dict(palavras_top)
        
        self.vocabulario = {
            palavra: idx 
            for idx, palavra in enumerate(sorted(vocab_filtrado.keys()))
        }
        
        self.idf = {}
        for palavra, idx in self.vocabulario.items():
            self.idf[idx] = math.log(num_docs / doc_freq[palavra])
        
        fim = time.perf_counter()
        self.tempo_construcao_vocab = fim - inicio
        
        return self.vocabulario, num_docs
    
    def calcular_tfidf(self, documentos_processados):
        inicio = time.perf_counter()
        
        linhas = []
        colunas = []
        valores = []
        
        for doc_idx, tokens_lematizados in enumerate(documentos_processados):
            if not tokens_lematizados:
                continue
            
            tf = Counter(tokens_lematizados)
            total_tokens_doc = len(tokens_lematizados)
            
            for palavra, freq in tf.items():
                if palavra in self.vocabulario:
                    idx_palavra = self.vocabulario[palavra]
                    tf_valor = freq / total_tokens_doc
                    tfidf_valor = tf_valor * self.idf[idx_palavra]
                    
                    linhas.append(doc_idx)
                    colunas.append(idx_palavra)
                    valores.append(np.float32(tfidf_valor))
            
            if doc_idx % 10000 == 0:
                memoria_atual = self.processo.memory_info().rss
                if memoria_atual > self.memoria_pico:
                    self.memoria_pico = memoria_atual
        
        matriz_tfidf = csr_matrix(
            (valores, (linhas, colunas)), 
            shape=(len(documentos_processados), len(self.vocabulario))
        )
        
        fim = time.perf_counter()
        self.tempo_vetorizacao = fim - inicio
        
        self.memoria_final = self.processo.memory_info().rss
        if self.memoria_final > self.memoria_pico:
            self.memoria_pico = self.memoria_final
        
        return len(documentos_processados), matriz_tfidf
    
    def calcular_hashes(self, documentos_processados, matriz_tfidf):
        hash_docs = hashlib.md5(str(documentos_processados).encode()).hexdigest()
        hash_vocab = hashlib.md5(str(sorted(self.vocabulario.items())).encode()).hexdigest()
        hash_matriz = hashlib.md5(matriz_tfidf.data.tobytes() + matriz_tfidf.indices.tobytes()).hexdigest()
        
        print("HASHES DE VALIDAÇÃO:")
        print(f"   Documentos:  {hash_docs}")
        print(f"   Vocabulário: {hash_vocab}")
        print(f"   Matriz:      {hash_matriz}")
        
        return hash_docs, hash_vocab, hash_matriz
    
    def estatisticas(self):
        if self.num_documentos > 0:
            print(f"Documentos processados: {self.num_documentos:,}")
            print(f"Total de tokens:        {self.total_tokens:,}")
            print("TEMPOS:")
            print(f"   Construção vocabulário: {self.tempo_construcao_vocab:.3f}s")
            print(f"   Tokenização:            {self.tempo_tokenizacao:.3f}s")
            print(f"   Lematização:            {self.tempo_lematizacao:.3f}s")
            print(f"   Vetorização TF-IDF:     {self.tempo_vetorizacao:.3f}s")
            
            tempo_total_nlp = self.tempo_tokenizacao + self.tempo_lematizacao + self.tempo_vetorizacao
            tempo_total_serial = self.tempo_construcao_vocab + tempo_total_nlp
            print(f"   TOTAL NLP (CPU):        {tempo_total_nlp:.3f}s")
            print(f"   TOTAL SERIAL:           {tempo_total_serial:.3f}s")
            print(f"   TOTAL EXECUÇÃO:         {self.tempo_execucao_total:.3f}s")
            
            print("MEMÓRIA:")
            memoria_inicial_mb = self.memoria_inicial / 1024**2
            memoria_pico_mb = self.memoria_pico / 1024**2
            memoria_final_mb = self.memoria_final / 1024**2
            memoria_consumida_mb = memoria_pico_mb - memoria_inicial_mb
            
            print(f"   Inicial:      {memoria_inicial_mb:.1f} MB")
            print(f"   Pico:         {memoria_pico_mb:.1f} MB") 
            print(f"   Final:        {memoria_final_mb:.1f} MB")
            print(f"   Consumida:    {memoria_consumida_mb:.1f} MB")
    
    def guardar_estatisticas(self, arquivo=None):
        if arquivo is None:
            arquivo = f'estatisticas_serial_{self.coluna}.txt'
            
        with open(arquivo, 'w', encoding='utf-8') as f:
            f.write(f"Documentos processados: {self.num_documentos:,}\n")
            f.write(f"Total de tokens:        {self.total_tokens:,}\n")
            f.write("\n")
            
            f.write("TEMPOS:\n")
            f.write(f"   Construção vocabulário: {self.tempo_construcao_vocab:.3f}s\n")
            f.write(f"   Tokenização:            {self.tempo_tokenizacao:.3f}s\n")
            f.write(f"   Lematização:            {self.tempo_lematizacao:.3f}s\n")
            f.write(f"   Vetorização TF-IDF:     {self.tempo_vetorizacao:.3f}s\n")
            
            tempo_total_nlp = self.tempo_tokenizacao + self.tempo_lematizacao + self.tempo_vetorizacao
            tempo_total_serial = self.tempo_construcao_vocab + tempo_total_nlp
            f.write(f"   TOTAL NLP (CPU):        {tempo_total_nlp:.3f}s\n")
            f.write(f"   TOTAL SERIAL:           {tempo_total_serial:.3f}s\n")
            f.write(f"   TOTAL EXECUÇÃO:         {self.tempo_execucao_total:.3f}s\n")
            f.write("\n")
            
            f.write("MEMÓRIA:\n")
            memoria_inicial_mb = self.memoria_inicial / 1024**2
            memoria_pico_mb = self.memoria_pico / 1024**2
            memoria_final_mb = self.memoria_final / 1024**2
            memoria_consumida_mb = memoria_pico_mb - memoria_inicial_mb
            
            f.write(f"   Inicial:      {memoria_inicial_mb:.1f} MB\n")
            f.write(f"   Pico:         {memoria_pico_mb:.1f} MB\n")
            f.write(f"   Final:        {memoria_final_mb:.1f} MB\n")
            f.write(f"   Consumida:    {memoria_consumida_mb:.1f} MB\n")

def processar_arquivo(caminho_arquivo, porcentagem=20, coluna='text'):
    
    # CRIA O MONITOR
    nome_dataset = Path(caminho_arquivo).stem
    arquivo_monitor = f'monitoring_logs/serial_{nome_dataset}_{porcentagem}p.txt'
    Path('monitoring_logs').mkdir(exist_ok=True)
    
    monitor = Monitor(nome_arquivo=arquivo_monitor, intervalo=60)
    monitor.iniciar()

    inicio_execucao_total = time.perf_counter()
    
    processador = processing(
        min_df=10,
        max_df=0.9,
        max_features=50000,
        porcentagem=porcentagem,
        coluna=coluna
    )
    
    monitor.etapa("01_Carregamento")
    textos = processador.carregar_dados(caminho_arquivo)
    processador.atualizar_pico()
    
    monitor.etapa("02_Tokenizacao_Lematizacao")
    documentos_processados = processador.processar_documentos(textos)
    processador.atualizar_pico()
    
    monitor.etapa("03_Construcao_Vocabulario")
    vocab, num_docs = processador.construir_vocabulario(documentos_processados)
    processador.atualizar_pico()
    
    monitor.etapa("04_Vetorizacao_TFIDF")
    total_processados, matriz_tfidf = processador.calcular_tfidf(documentos_processados)

    fim_execucao_total = time.perf_counter()
    processador.tempo_execucao_total = fim_execucao_total - inicio_execucao_total
    
    monitor.etapa("05_Finalizacao")
    processador.calcular_hashes(documentos_processados, matriz_tfidf)
    processador.estatisticas()
    processador.guardar_estatisticas()
    
    monitor.parar()
    
    return processador, matriz_tfidf

if __name__ == "__main__":
    import sys
    
    arquivo = sys.argv[1] if len(sys.argv) > 1 else "amazon.csv"
    porcentagem = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    coluna = sys.argv[3] if len(sys.argv) > 3 else 'text'
    
    processador, matriz = processar_arquivo(arquivo, porcentagem, coluna)