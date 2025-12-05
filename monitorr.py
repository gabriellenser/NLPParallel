import time
import psutil
import threading
from datetime import datetime

class Monitorr:
    
    def __init__(self, nome_arquivo, intervalo=60):
        self.intervalo = intervalo
        self.processo = psutil.Process()
        self.rodando = False
        self.thread = None
        
        self.memorias = []
        self.cpus = []
        self.tempos = []
        self.etapas_log = []
        self.etapa_atual = None
        self.inicio_etapa = None
        self.inicio_total = time.time()
        
        self.arquivo = nome_arquivo
    
    def iniciar(self):
        self.rodando = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def _loop(self):
        while self.rodando:
            tempo = time.time() - self.inicio_total
            mem = self.processo.memory_info().rss / 1024**2
            cpu = self.processo.cpu_percent(interval=0.1)
            
            self.tempos.append(tempo)
            self.memorias.append(mem)
            self.cpus.append(cpu)
            
            time.sleep(self.intervalo)
    
    def etapa(self, nome):
        if self.etapa_atual:
            duracao = time.time() - self.inicio_etapa
            mem = self.processo.memory_info().rss / 1024**2
            self.etapas_log.append({
                'nome': self.etapa_atual,
                'duracao': duracao,
                'memoria_mb': mem
            })
        
        self.etapa_atual = nome
        self.inicio_etapa = time.time()
    
    def parar(self):
        if self.etapa_atual:
            duracao = time.time() - self.inicio_etapa
            mem = self.processo.memory_info().rss / 1024**2
            self.etapas_log.append({
                'nome': self.etapa_atual,
                'duracao': duracao,
                'memoria_mb': mem
            })
        
        self.rodando = False
        if self.thread:
            self.thread.join(timeout=2)
        
        self._salvar()
    
    def _salvar(self):
        with open(self.arquivo, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE MONITORAMENTO")
            f.write("="*80 + "\n\n")
            
            duracao = time.time() - self.inicio_total
            f.write(f"Duração total: {duracao:.2f}s ({duracao/60:.2f}min)\n")
            f.write(f"Amostras: {len(self.memorias)}\n")
            f.write(f"Intervalo: {self.intervalo}s\n\n")
            
            if self.memorias:
                f.write("MEMÓRIA:\n")
                f.write(f"  Máxima: {max(self.memorias):.1f} MB\n")
                f.write(f"  Média:  {sum(self.memorias)/len(self.memorias):.1f} MB\n")
                f.write(f"  Mínima: {min(self.memorias):.1f} MB\n\n")
            
            if self.etapas_log:
                f.write("ETAPAS:\n")
                f.write(f"{'Nome':<35} | {'Duração':>10} | {'Memória':>10}\n")
                f.write("-"*80 + "\n")
                for e in self.etapas_log:
                    f.write(f"{e['nome']:<35} | {e['duracao']:9.2f}s | {e['memoria_mb']:9.1f}MB\n")
                f.write("-"*80 + "\n")
                f.write(f"{'TOTAL':<35} | {sum(x['duracao'] for x in self.etapas_log):9.2f}s |\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DADOS BRUTOS (tempo_min, memoria_mb, cpu_percent, etapa)\n")
            f.write("="*80 + "\n")
            
            for t, m, c in zip(self.tempos, self.memorias, self.cpus):
                
                etapa_nome = "inicial"
                tempo_acumulado = 0
                for etapa in self.etapas_log:
                    if tempo_acumulado <= t < tempo_acumulado + etapa['duracao']:
                        etapa_nome = etapa['nome']
                        break
                    tempo_acumulado += etapa['duracao']
                
                f.write(f"{t/60:.3f}, {m:.1f}, {c:.1f}, {etapa_nome}\n")