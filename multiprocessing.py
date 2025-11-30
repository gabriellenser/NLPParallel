import csv
import re
import math
import time
import signal
import sys
import hashlib
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer
import nltk
from scipy.sparse import csr_matrix
import numpy as np
import psutil
import multiprocessing as mp

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def inicializar_worker():
    global lemmatizer_local
    lemmatizer_local = WordNetLemmatizer()

def tokenizar_texto(texto):
    if not texto:
        return []
    texto_lower = texto.lower()
    return re.findall(r'\b[a-z0-9]+\b', texto_lower)

def processar_lote(lote_textos):
    import time
    
    inicio_tokenizacao = time.perf_counter()
    todos_tokens = []
    total_tokens_lote = 0
    
    for texto in lote_textos:
        tokens = tokenizar_texto(texto)
        todos_tokens.append(tokens)
        total_tokens_lote += len(tokens)
    
    fim_tokenizacao = time.perf_counter()
    tempo_tokenizacao = fim_tokenizacao - inicio_tokenizacao
    
    inicio_lematizacao = time.perf_counter()
    documentos_processados = []
    
    for tokens in todos_tokens:
        if tokens:
            tokens_lematizados = [lemmatizer_local.lemmatize(token) for token in tokens]
            documentos_processados.append(tokens_lematizados)
        else:
            documentos_processados.append([])
    
    fim_lematizacao = time.perf_counter()
    tempo_lematizacao = fim_lematizacao - inicio_lematizacao
    
    return documentos_processados, total_tokens_lote, tempo_tokenizacao, tempo_lematizacao

def construir_vocab_lote(lote_documentos):
    doc_freq_local = defaultdict(int)
    num_docs_local = 0
    
    for tokens_lematizados in lote_documentos:
        if tokens_lematizados:
            num_docs_local += 1
            for palavra in set(tokens_lematizados):
                doc_freq_local[palavra] += 1
    
    return doc_freq_local, num_docs_local

def calcular_tfidf_lote(args):
    documentos_lematizados, vocabulario, idf, doc_offset = args
    
    linhas = []
    colunas = []
    valores = []
    
    for doc_local_idx, tokens_lematizados in enumerate(documentos_lematizados):
        if not tokens_lematizados:
            continue
            
        tf = Counter(tokens_lematizados)
        total_tokens_doc = len(tokens_lematizados)
        doc_global_idx = doc_offset + doc_local_idx
        
        for palavra, freq in tf.items():
            if palavra in vocabulario:
                idx_palavra = vocabulario[palavra]
                tf_valor = freq / total_tokens_doc
                tfidf_valor = tf_valor * idf[idx_palavra]
                
                linhas.append(doc_global_idx)
                colunas.append(idx_palavra)
                valores.append(np.float32(tfidf_valor))
    
    return (
        np.array(linhas, dtype=np.int32),
        np.array(colunas, dtype=np.int32), 
        np.array(valores, dtype=np.float32)
    )


class processing:
    
    def __init__(self, min_df=10, max_df=0.9, max_features=50000, porcentagem=100, num_processes=None, coluna='text'):
        self.lemmatizer = WordNetLemmatizer()
        
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.porcentagem = porcentagem
        self.num_processes = num_processes or mp.cpu_count()
        self.coluna = coluna
        
        self.vocabulario = {}
        self.idf = {}
        
        self.tempo_construcao_vocab = 0.0
        self.tempo_tokenizacao = 0.0
        self.tempo_lematizacao = 0.0
        self.tempo_vetorizacao = 0.0
        self.tempo_consolidacao = 0.0
        self.tempo_consolidacao_docs = 0.0
        self.tempo_consolidacao_vocab = 0.0
        self.tempo_execucao_total = 0.0
        self.num_documentos = 0
        self.total_tokens = 0
        
        self.processo = psutil.Process()
        self.memoria_inicial = 0
        self.memoria_pico = 0
        self.memoria_final = 0
        
        self.pool = None
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def atualizar_pico(self):
        memoria_atual = self.processo.memory_info().rss
        if memoria_atual > self.memoria_pico:
            self.memoria_pico = memoria_atual
    
    def _signal_handler(self, signum, frame):
        print(f"\nEncerrando processos")
        self._cleanup_pool()
        print("Saindo")
        sys.exit(0)
    
    def _cleanup_pool(self):
        if self.pool:
            try:
                self.pool.close()  
                self.pool.join(timeout=3)  
                print("Pool finalizada")
            except Exception as e:
                print(f"Forçanco encerramento {e}")
                try:
                    self.pool.terminate()  
                    self.pool.join(timeout=2)  
                    print("Pool terminado")
                except Exception as e2:
                    print(f"Erro no termino: {e2}")
            finally:
                self.pool = None
    
    def __enter__(self):
        try:
            self.pool = mp.Pool(self.num_processes, initializer=inicializar_worker)
            return self
        except Exception as e:
            print(f"Erro ao criar pool: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_pool()
        
        if exc_type == KeyboardInterrupt:
            print("\nEstatísticas parciais.")
            if self.num_documentos > 0:
                self.estatisticas()
                self.salvar_dados()
    
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
    
    def dividir_em_lotes(self, dados, tamanho_lote):
        for i in range(0, len(dados), tamanho_lote):
            yield dados[i:i + tamanho_lote]
    
    def carregar_dados(self, caminho_arquivo):
        max_docs = self.linhas_para_processar(caminho_arquivo)
        self.memoria_inicial = self.processo.memory_info().rss
        self.memoria_pico = self.memoria_inicial
        
        textos = list(self.ler_csv(caminho_arquivo, max_lines=max_docs))
        
        return textos
    
    def processar_documentos(self, textos):
        
        tamanho_lote = len(textos) // self.num_processes
        lotes = list(self.dividir_em_lotes(textos, max(tamanho_lote, 1)))
        
        try:
            resultados_completos = self.pool.map(processar_lote, lotes)
        except KeyboardInterrupt:
            raise
        
        inicio_consolidacao_docs = time.perf_counter()
        
        todos_documentos = []
        total_tokens = 0
        tempos_tokenizacao = []
        tempos_lematizacao = []
        
        for documentos_lote, tokens_count, tempo_tok, tempo_lem in resultados_completos:
            todos_documentos.extend(documentos_lote)
            total_tokens += tokens_count
            tempos_tokenizacao.append(tempo_tok)
            tempos_lematizacao.append(tempo_lem)
        
        fim_consolidacao_docs = time.perf_counter()
        self.tempo_consolidacao_docs = fim_consolidacao_docs - inicio_consolidacao_docs
        
        self.tempo_tokenizacao = max(tempos_tokenizacao)  
        self.tempo_lematizacao = max(tempos_lematizacao)  
        self.total_tokens = total_tokens
        self.num_documentos = len(todos_documentos)
        
        return todos_documentos
    
    def construir_vocabulario(self, documentos_processados):
        inicio = time.perf_counter()
        
        docs_por_processo = len(documentos_processados) // self.num_processes
        lotes_docs = []
        
        for i in range(self.num_processes):
            inicio_lote = i * docs_por_processo
            if i == self.num_processes - 1:
                fim_lote = len(documentos_processados)
            else:
                fim_lote = (i + 1) * docs_por_processo
            
            lote = documentos_processados[inicio_lote:fim_lote]
            lotes_docs.append(lote)
        
        try:
            resultados = self.pool.map(construir_vocab_lote, lotes_docs)
        except KeyboardInterrupt:
            raise
        
        inicio_consolidacao_vocab = time.perf_counter()
        
        doc_freq = defaultdict(int)
        num_docs_total = 0
        
        for doc_freq_local, num_docs_local in resultados:
            for palavra, freq in doc_freq_local.items():
                doc_freq[palavra] += freq
            num_docs_total += num_docs_local
        
        fim_consolidacao_vocab = time.perf_counter()
        self.tempo_consolidacao_vocab = fim_consolidacao_vocab - inicio_consolidacao_vocab
        
        max_doc_freq = int(self.max_df * num_docs_total)
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
            self.idf[idx] = math.log(num_docs_total / doc_freq[palavra])
        
        fim = time.perf_counter()
        self.tempo_construcao_vocab = fim - inicio
        
        return self.vocabulario, num_docs_total
    
    def calcular_tfidf(self, documentos_processados):
        inicio = time.perf_counter()
        
        docs_por_processo = len(documentos_processados) // self.num_processes
        lotes_args = []
        
        for i in range(self.num_processes):
            inicio_lote = i * docs_por_processo
            if i == self.num_processes - 1:
                fim_lote = len(documentos_processados)
            else:
                fim_lote = (i + 1) * docs_por_processo
            
            lote_docs = documentos_processados[inicio_lote:fim_lote]
            lotes_args.append((lote_docs, self.vocabulario, self.idf, inicio_lote))
        
        try:
            resultados_tfidf = self.pool.map(calcular_tfidf_lote, lotes_args)
        except KeyboardInterrupt:
            raise
        
        fim = time.perf_counter()
        self.tempo_vetorizacao = fim - inicio
        
        inicio_consolidacao = time.perf_counter()
        
        if resultados_tfidf:
            linhas_arrays = [linhas for linhas, _, _ in resultados_tfidf if len(linhas) > 0]
            colunas_arrays = [colunas for _, colunas, _ in resultados_tfidf if len(colunas) > 0]
            valores_arrays = [valores for _, _, valores in resultados_tfidf if len(valores) > 0]
            
            linhas_final = np.concatenate(linhas_arrays) if linhas_arrays else np.array([], dtype=np.int32)
            colunas_final = np.concatenate(colunas_arrays) if colunas_arrays else np.array([], dtype=np.int32)
            valores_final = np.concatenate(valores_arrays) if valores_arrays else np.array([], dtype=np.float32)
        else:
            linhas_final = np.array([], dtype=np.int32)
            colunas_final = np.array([], dtype=np.int32)
            valores_final = np.array([], dtype=np.float32)
        
        matriz_tfidf = csr_matrix(
            (valores_final, (linhas_final, colunas_final)), 
            shape=(len(documentos_processados), len(self.vocabulario))
        )
        
        fim_consolidacao = time.perf_counter()
        self.tempo_consolidacao = fim_consolidacao - inicio_consolidacao
        
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
            print(f"   Consolidação Docs:      {self.tempo_consolidacao_docs:.3f}s")
            print(f"   Consolidação Vocab:     {self.tempo_consolidacao_vocab:.3f}s")
            print(f"   Consolidação TF-IDF:    {self.tempo_consolidacao:.3f}s")
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
    
    def salvar_dados(self, arquivo=None):
        if arquivo is None:
            arquivo = f'estatisticas_paralelo_{self.coluna}.txt'
            
        with open(arquivo, 'w', encoding='utf-8') as f:
            f.write(f"Documentos processados: {self.num_documentos:,}\n")
            f.write(f"Total de tokens:        {self.total_tokens:,}\n")
            f.write("\n")
            
            f.write("TEMPOS:\n")
            f.write(f"   Construção vocabulário: {self.tempo_construcao_vocab:.3f}s\n")
            f.write(f"   Tokenização:            {self.tempo_tokenizacao:.3f}s\n")
            f.write(f"   Lematização:            {self.tempo_lematizacao:.3f}s\n")
            f.write(f"   Vetorização TF-IDF:     {self.tempo_vetorizacao:.3f}s\n")
            f.write(f"   Consolidação Docs:      {self.tempo_consolidacao_docs:.3f}s\n")
            f.write(f"   Consolidação Vocab:     {self.tempo_consolidacao_vocab:.3f}s\n")
            f.write(f"   Consolidação TF-IDF:    {self.tempo_consolidacao:.3f}s\n")
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

def processar_arquivo(caminho_arquivo, porcentagem=20, num_processes=None, coluna='text'):
    
    inicio_execucao_total = time.perf_counter()
    
    with processing(
        min_df=10,
        max_df=0.9,
        max_features=50000,
        porcentagem=porcentagem,
        num_processes=num_processes,
        coluna=coluna
    ) as processador:
        
        textos = processador.carregar_dados(caminho_arquivo)
        processador.atualizar_pico()
        
        documentos_processados = processador.processar_documentos(textos)
        processador.atualizar_pico()
        
        vocab, num_docs = processador.construir_vocabulario(documentos_processados)
        processador.atualizar_pico()
        
        total_processados, matriz_tfidf = processador.calcular_tfidf(documentos_processados)
        
        fim_execucao_total = time.perf_counter()
        processador.tempo_execucao_total = fim_execucao_total - inicio_execucao_total
        
        processador.calcular_hashes(documentos_processados, matriz_tfidf)
        processador.estatisticas()
        processador.salvar_dados()
        
        return processador, matriz_tfidf

if __name__ == "__main__":
    import sys
    
    arquivo = sys.argv[1] if len(sys.argv) > 1 else "amazon.csv"
    porcentagem = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    num_processes = int(sys.argv[3]) if len(sys.argv) > 3 else None
    coluna = sys.argv[4] if len(sys.argv) > 4 else 'text'
    
    processador, matriz = processar_arquivo(arquivo, porcentagem, num_processes, coluna)