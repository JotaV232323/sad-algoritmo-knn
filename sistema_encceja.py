import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# 1. Configura diretório
NOME_ARQUIVO = "MICRODADOS_ENCCEJA_2024_REG_NAC.csv"
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_CSV = os.path.join(PASTA_DO_SCRIPT, NOME_ARQUIVO)

# ==========================================
# 1. CARREGAMENTO E FILTRAGEM DOS DADOS
# ==========================================
def carregar_dados_reais():
    if not os.path.exists(ARQUIVO_CSV):
        print(f"ERRO CRÍTICO: O arquivo {ARQUIVO_CSV} não foi encontrado!")
        return None

    print("Iniciando ETL do arquivo.")
    
    # Colunas importantes
    colunas_necessarias = [
        'TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_CERTIFICACAO', 'Q26',
        'NU_NOTA_CH', 'NU_NOTA_MT', 'NU_NOTA_LC', 'NU_NOTA_CN', 'NU_NOTA_REDACAO'
    ]
    
    amostras = []
    total_acumulado = 0
    limite_registros = 200000
    
    # Lê o arquivo em blocos de linhas
    for chunk in pd.read_csv(ARQUIVO_CSV, sep=';', encoding='latin-1', chunksize=20000):
        # Nomes das colunas em maiúsculo
        chunk.columns = chunk.columns.str.upper()
        
        # Captura colunas importantes
        chunk_filtrado = chunk[colunas_necessarias]
        
        # Remove Nulos em qualquer uma das provas
        chunk_limpo = chunk_filtrado.dropna(subset=['NU_NOTA_CH', 'NU_NOTA_MT', 'NU_NOTA_LC', 'NU_NOTA_CN', 'NU_NOTA_REDACAO', 'TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_CERTIFICACAO', 'Q26'])
        
        # Descarta inconsistências
        chunk_limpo = chunk_limpo[chunk_limpo['TP_SEXO'].isin(['M', 'F', 'm', 'f'])]
        chunk_limpo = chunk_limpo[chunk_limpo['Q26'].isin(['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd'])]
        
        if len(chunk_limpo) > 0:
            amostras.append(chunk_limpo)
            total_acumulado += len(chunk_limpo)
            print(f"Filtrados com sucesso: {total_acumulado}")
            
        if total_acumulado >= limite_registros:
            break
            
    if total_acumulado == 0:
        print("\nERRO: Nenhum aluno com notas e informações válidas")
        return None
        
    df = pd.concat(amostras).head(limite_registros)
    
    # Mapeando 'M' para 0 e 'F' para 1
    df['TP_SEXO'] = df['TP_SEXO'].map({'M': 0, 'F': 1, 'm': 0, 'f': 1})
    
    # Converte as letras em números para a IA calcular
    df['Q26'] = df['Q26'].str.upper().map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
    
    print(f"\nCONCLUÍDO: Base real com {len(df)} estudantes")
    return df

# Inicialização dos dados e treinamento da IA
df_real = carregar_dados_reais()

if df_real is None:
    messagebox.showerror("Erro", "Falha ao processar. Verifique o arquivo.")
    exit()

# Características de entrada para a IA
X_dados = df_real[['TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_CERTIFICACAO', 'Q26']]
Y_notas = df_real[['NU_NOTA_CH', 'NU_NOTA_MT', 'NU_NOTA_LC', 'NU_NOTA_CN', 'NU_NOTA_REDACAO']]

# Aplicação da Normalização Padrão Z-Score
scaler = StandardScaler()
X_dados_scaled = scaler.fit_transform(X_dados)

# Cria e treina o KNN com K=5 vizinhos
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_dados_scaled, Y_notas)

# ==========================================
# 2. INTERFACE GRÁFICA
# ==========================================
def processar_decisao():
    try:
        # Traduz a faixa etária selecionada para o código numérico do INEP
        idade_texto = combo_idade.get()
        if "Menor de 17" in idade_texto: faixa_etaria = 1
        elif "17 anos" in idade_texto: faixa_etaria = 2
        elif "18 anos" in idade_texto: faixa_etaria = 3
        elif "19 anos" in idade_texto: faixa_etaria = 4
        elif "20 anos" in idade_texto: faixa_etaria = 5
        elif "21 anos" in idade_texto: faixa_etaria = 6
        elif "22 anos" in idade_texto: faixa_etaria = 7
        elif "23 anos" in idade_texto: faixa_etaria = 8
        elif "24 anos" in idade_texto: faixa_etaria = 9
        elif "25 anos" in idade_texto: faixa_etaria = 10
        elif "26 e 30" in idade_texto: faixa_etaria = 11
        elif "31 e 35" in idade_texto: faixa_etaria = 12
        elif "36 e 40" in idade_texto: faixa_etaria = 13
        elif "41 e 45" in idade_texto: faixa_etaria = 14
        else: faixa_etaria = 15 # Acima de 46 anos
        
        # Traduz os parâmetros de tela para a base
        sexo = 1 if combo_sexo.get() == "Feminino" else 0
        trabalha = 1 if combo_trabalha.get() == "Sim" else 0
        certificacao = 2 if combo_cert.get() == "Ensino Médio" else 1
    
        dic_renda = {"Até 1 Salário Mínimo": 1, "Entre 1 e 2 Salários": 2, "Entre 2 e 3 Salários": 3, "Mais de 3 Salários": 4}
        renda = dic_renda[combo_renda.get()]
        
        # Formata os dados do aluno novo e aplica a mesma normalização
        aluno_novo = pd.DataFrame([[faixa_etaria, sexo, certificacao, renda]], columns=X_dados.columns)
        aluno_novo_scaled = scaler.transform(aluno_novo)
        
        # 1. Prevê as notas do aluno novo via KNN
        notas_previstas = knn.predict(aluno_novo_scaled)[0]
        nota_ch_p, nota_mat_p, nota_lc_p, nota_cn_p, nota_red_p = notas_previstas[0], notas_previstas[1], notas_previstas[2], notas_previstas[3], notas_previstas[4]
        
        # 2. Identifica os 5 Vizinhos Reais do INEP para Comparação
        distancias, indices = knn.kneighbors(aluno_novo_scaled)
        vizinhos = df_real.iloc[indices[0]]
        
        # Extração das notas individuais de cada vizinho
        notas_ch_vizinhos = vizinhos['NU_NOTA_CH'].values
        notas_mat_vizinhos = vizinhos['NU_NOTA_MT'].values
        notas_lc_vizinhos = vizinhos['NU_NOTA_LC'].values
        notas_cn_vizinhos = vizinhos['NU_NOTA_CN'].values
        notas_red_vizinhos = vizinhos['NU_NOTA_REDACAO'].values
        
        # LIMPA E PREENCHE A TABELA VISUAL DE VIZINHOS COM STATUS DE APROVAÇÃO E CORES
        for item in tabela_vizinhos.get_children():
            tabela_vizinhos.delete(item)
            
        for i in range(5):
            # Regra do INEP: Passa se tirar >= 100 em TODAS as objetivas e >= 5.0 na redação
            passou_geral = (notas_ch_vizinhos[i] >= 100 and notas_mat_vizinhos[i] >= 100 and 
                            notas_lc_vizinhos[i] >= 100 and notas_cn_vizinhos[i] >= 100 and 
                            notas_red_vizinhos[i] >= 5.0)
            status_texto = "APROVADO" if passou_geral else "REPROVADO"
            tag_cor = "aprovado_linha" if passou_geral else "reprovado_linha"
            
            tabela_vizinhos.insert("", tk.END, values=(
                f"Vizinho {i+1}", 
                f"{notas_ch_vizinhos[i]:.1f}", 
                f"{notas_mat_vizinhos[i]:.1f}", 
                f"{notas_lc_vizinhos[i]:.1f}", 
                f"{notas_cn_vizinhos[i]:.1f}", 
                f"{notas_red_vizinhos[i]:.1f}",
                status_texto
            ), tags=(tag_cor,))
        
        # Conta quantos dos 5 vizinhos foram reprovados por matéria
        reprovados_ch = sum(1 for n in notas_ch_vizinhos if n < 100)
        reprovados_mat = sum(1 for n in notas_mat_vizinhos if n < 100)
        reprovados_lc = sum(1 for n in notas_lc_vizinhos if n < 100)
        reprovados_cn = sum(1 for n in notas_cn_vizinhos if n < 100)
        reprovados_red = sum(1 for n in notas_red_vizinhos if n < 5.0)
        
        # Conta quantos vizinhos individuais ficaram acima da nota
        # Ciências Humanas
        vizinhos_acima_ch = sum(1 for n in notas_ch_vizinhos if n > nota_ch_p)
        if vizinhos_acima_ch >= 3:
            comparacao_ch = "sua nota INFERIOR à maioria deles."
        else:
            comparacao_ch = "sua nota SUPERIOR à maioria deles."

        # Matemática
        vizinhos_acima_mat = sum(1 for n in notas_mat_vizinhos if n > nota_mat_p)
        if vizinhos_acima_mat >= 3:
            comparacao_mat = "sua nota INFERIOR à maioria deles."
        else:
            comparacao_mat = "sua nota SUPERIOR à maioria deles."

        # Linguagens e Códigos
        vizinhos_acima_lc = sum(1 for n in notas_lc_vizinhos if n > nota_lc_p)
        if vizinhos_acima_lc >= 3:
            comparacao_lc = "sua nota INFERIOR à maioria deles."
        else:
            comparacao_lc = "sua nota SUPERIOR à maioria deles."
        
        # Ciências da Natureza
        vizinhos_acima_cn = sum(1 for n in notas_cn_vizinhos if n > nota_cn_p)
        if vizinhos_acima_cn >= 3:
            comparacao_cn = "sua nota INFERIOR à maioria deles."
        else:
            comparacao_cn = "sua nota SUPERIOR à maioria deles."

        # Redação
        vizinhos_acima_red = sum(1 for n in notas_red_vizinhos if n > nota_red_p)
        if vizinhos_acima_red >= 3:
            comparacao_red = "sua nota INFERIOR à maioria deles."
        else:
            comparacao_red = "sua nota SUPERIOR à maioria deles."
        
        # Lógica de Tomada de Decisão Pedagógica
        recomendacoes = []
        risco_reprovacao = False
        
        # 1. Alerta de Ciências Humanas
        if nota_ch_p < 100:
            recomendacoes.append(f"REFORÇO EM CIÊNCIAS HUMANAS: Nota prevista ({nota_ch_p:.1f}) exige atenção preventiva.")
            risco_reprovacao = True
        else:
            recomendacoes.append(f"CIÊNCIAS HUMANAS ESTÁVEL: Projeção de aprovação regular ({nota_ch_p:.1f}), sendo {comparacao_ch}")
        
        if reprovados_ch >= 3:
            risco_reprovacao = True
            recomendacoes.append(f"ALERTA EM CIÊNCIAS HUMANAS: {reprovados_ch} vizinhos do mesmo perfil socioeconômico foram reprovados, sendo {comparacao_ch}")

        # 2. Alerta de Matemática
        if nota_mat_p < 100:
            recomendacoes.append(f"REFORÇO EM MATEMÁTICA: Nota prevista ({nota_mat_p:.1f}) exige atenção preventiva.")
            risco_reprovacao = True
        else:
            recomendacoes.append(f"MATEMÁTICA ESTÁVEL: Projeção de aprovação regular ({nota_mat_p:.1f}), sendo {comparacao_mat}")

        if reprovados_mat >= 3:
            risco_reprovacao = True
            recomendacoes.append(f"ALERTA EM MATEMÁTICA: {reprovados_mat} vizinhos do mesmo perfil socioeconômico foram reprovados, sendo {comparacao_mat}")

        # 3. Alerta de Linguagens
        if nota_lc_p < 100 or reprovados_lc >= 3:
            recomendacoes.append(f"REFORÇO EM LINGUAGENS: Nota prevista ({nota_lc_p:.1f}) exige atenção preventiva.")
            risco_reprovacao = True
        else:
            recomendacoes.append(f"LINGUAGENS ESTÁVEL: Projeção de aprovação regular ({nota_lc_p:.1f}), sendo {comparacao_lc}")
        
        if reprovados_lc >= 3:
            risco_reprovacao = True
            recomendacoes.append(f"ALERTA EM LINGUAGENS: {reprovados_lc} vizinhos do mesmo perfil socioeconômico foram reprovados, sendo {comparacao_lc}")
        
        # 4. Alerta de Ciências da Natureza
        if nota_cn_p < 100:
            recomendacoes.append(f"REFORÇO EM CIÊNCIAS DA NATUREZA: Nota prevista ({nota_cn_p:.1f}) exige atenção preventiva.")
            risco_reprovacao = True
        else:
            recomendacoes.append(f"CIÊNCIAS DA NATUREZA ESTÁVEL: Projeção de aprovação regular ({nota_cn_p:.1f}), sendo {comparacao_cn}")

        if reprovados_cn >= 3:
            risco_reprovacao = True
            recomendacoes.append(f"ALERTA EM CIÊNCIAS DA NATUREZA: {reprovados_cn} vizinhos do mesmo perfil socioeconômico foram reprovados, sendo {comparacao_cn}")

        # 5. Alerta de Redação
        if nota_red_p < 5.0:
            recomendacoes.append(f"OFICINA DE REDAÇÃO OBRIGATÓRIA: Nota prevista ({nota_red_p:.1f}) exige atenção preventiva.")
            risco_reprovacao = True
        else:
            recomendacoes.append(f"REDAÇÃO ESTÁVEL: Projeção de aprovação regular ({nota_red_p:.1f}), sendo {comparacao_red}")

        if reprovados_red >= 3:
            risco_reprovacao = True
            recomendacoes.append(f"ALERTA EM REDAÇÃO: {reprovados_red} vizinhos do mesmo perfil socioeconômico foram reprovados, sendo o aluno {comparacao_red}")
            
        # Alerta estrutural de trabalho
        if trabalha == 1:
            recomendacoes.append("TURMA NOTURNA OU EAD. Recomenda-se cronograma flexível de estudos e materiais gravados.")
            
        # Diagnóstico Geral Gerencial
        if not risco_reprovacao:
            recomendacoes.append("ACOMPANHAMENTO PADRÃO: Aluno apresenta excelente projeção em todas as disciplinas. Focar em simulados de manutenção de desempenho.")
        else:
            recomendacoes.append(f"ALERTA PEDAGÓGICO GERAL: Alto risco de reprovação baseado em cruzamentos socioeconômicos do passado (Vizinhos reprovados -> CH: {reprovados_ch} | MAT: {reprovados_mat} | LC: {reprovados_lc} | CN: {reprovados_cn} | RED: {reprovados_red}).")

        # ATUALIZA OS TEXTOS E APLICA AS CORES CONDICIONAIS (Verde: #056608 | Vermelho: #8B0000)
        lbl_nota_ch.config(text=f"CH: {nota_ch_p:.1f}", fg="#056608" if nota_ch_p >= 100 else "#8B0000")
        lbl_nota_mat.config(text=f"MAT: {nota_mat_p:.1f}", fg="#056608" if nota_mat_p >= 100 else "#8B0000")
        lbl_nota_lc.config(text=f"LC: {nota_lc_p:.1f}", fg="#056608" if nota_lc_p >= 100 else "#8B0000")
        lbl_nota_cn.config(text=f"CN: {nota_cn_p:.1f}", fg="#056608" if nota_cn_p >= 100 else "#8B0000")
        lbl_nota_red.config(text=f"RED: {nota_red_p:.1f}", fg="#056608" if nota_red_p >= 5.0 else "#8B0000")
        
        # Atualiza a caixa de recomendações textuais
        txt_recomendacoes.delete('1.0', tk.END)
        for rec in recomendacoes:
            txt_recomendacoes.insert(tk.END, f"{rec}\n\n")
            
    except Exception as e:
        messagebox.showerror("Erro de Entrada", f"Por favor, preencha todos os campos corretamente: {str(e)}")

# Configuração da Janela Principal
janela = tk.Tk()
janela.title("SAD ENCCEJA - Base de Dados Oficial INEP")
janela.geometry("740x820")
janela.configure(bg="#f4f6f9")
janela.attributes('-topmost', True)

lbl_titulo = tk.Label(janela, text="Apoio à Decisão Educacional (Algoritmo KNN)", font=("Arial", 16, "bold"), bg="#f4f6f9", fg="#2c3e50")
lbl_titulo.pack(pady=10)

frame_form = tk.LabelFrame(janela, text=" Dados de Matrícula do Novo Aluno ", font=("Arial", 11, "bold"), bg="#f4f6f9", pady=5, padx=10)
frame_form.pack(pady=5, fill="x", padx=20)

tk.Label(frame_form, text="Faixa Etária:", bg="#f4f6f9").grid(row=0, column=0, sticky="w", pady=5)
combo_idade = ttk.Combobox(frame_form, values=["Menor de 17 anos", "17 anos", "18 anos", "19 anos", "20 anos", "21 anos", "22 anos", "23 anos", "24 anos", "25 anos", "Entre 26 e 30 anos", "Entre 31 e 35 anos", "Entre 36 e 40 anos", "Entre 41 e 45 anos", "Mais de 46 anos"], width=15, state="readonly")
combo_idade.current(2)
combo_idade.grid(row=0, column=1, pady=5, padx=5)

tk.Label(frame_form, text="Sexo:", bg="#f4f6f9").grid(row=0, column=2, sticky="w", pady=5)
combo_sexo = ttk.Combobox(frame_form, values=["Masculino", "Feminino"], width=15, state="readonly")
combo_sexo.current(0)
combo_sexo.grid(row=0, column=3, pady=5, padx=5)

tk.Label(frame_form, text="Situação de Trabalho:", bg="#f4f6f9").grid(row=1, column=0, sticky="w", pady=5)
combo_trabalha = ttk.Combobox(frame_form, values=["Não Trabalha", "Sim"], width=15, state="readonly")
combo_trabalha.current(0)
combo_trabalha.grid(row=1, column=1, pady=5, padx=5)

tk.Label(frame_form, text="Certificação Pretendida:", bg="#f4f6f9").grid(row=1, column=2, sticky="w", pady=5)
combo_cert = ttk.Combobox(frame_form, values=["Ensino Fundamental", "Ensino Médio"], width=15, state="readonly")
combo_cert.current(1)
combo_cert.grid(row=1, column=3, pady=5, padx=5)

tk.Label(frame_form, text="Faixa de Renda Familiar:", bg="#f4f6f9").grid(row=2, column=0, sticky="w", pady=5)
combo_renda = ttk.Combobox(frame_form, values=["Até 1 Salário Mínimo", "Entre 1 e 2 Salários", "Entre 2 e 3 Salários", "Mais de 3 Salários"], width=25, state="readonly")
combo_renda.current(0)
combo_renda.grid(row=2, column=1, columnspan=2, pady=5, padx=5, sticky="w")

btn_analisar = tk.Button(janela, text="PROCESSAR DECISÃO PEDAGÓGICA", font=("Arial", 11, "bold"), bg="#27ae60", fg="white", command=processar_decisao, height=2)
btn_analisar.pack(pady=5, fill="x", padx=20)

frame_res = tk.LabelFrame(janela, text=" Diagnóstico do Sistema ", font=("Arial", 11, "bold"), bg="#f4f6f9", pady=5, padx=10)
frame_res.pack(pady=5, fill="both", expand=True, padx=20)

tk.Label(frame_res, text="NOTAS PREVISTAS DO NOVO ALUNO:", font=("Arial", 10, "bold"), bg="#f4f6f9", fg="#2980b9").pack(anchor="w")
frame_notas_cand = tk.Frame(frame_res, bg="#f4f6f9")
frame_notas_cand.pack(anchor="w", pady=2)
lbl_nota_ch = tk.Label(frame_notas_cand, text="CH: --", font=("Arial", 10, "bold"), bg="#f4f6f9")
lbl_nota_ch.pack(side="left", padx=(0, 15))
lbl_nota_mat = tk.Label(frame_notas_cand, text="MAT: --", font=("Arial", 10, "bold"), bg="#f4f6f9")
lbl_nota_mat.pack(side="left", padx=15)
lbl_nota_lc = tk.Label(frame_notas_cand, text="LC: --", font=("Arial", 10, "bold"), bg="#f4f6f9")
lbl_nota_lc.pack(side="left", padx=15)
lbl_nota_cn = tk.Label(frame_notas_cand, text="CN: --", font=("Arial", 10, "bold"), bg="#f4f6f9")
lbl_nota_cn.pack(side="left", padx=15)
lbl_nota_red = tk.Label(frame_notas_cand, text="RED: --", font=("Arial", 10, "bold"), bg="#f4f6f9")
lbl_nota_red.pack(side="left", padx=15)

tk.Label(frame_res, text="BOLETIM INDIVIDUAL DOS 5 VIZINHOS MAIS PRÓXIMOS:", font=("Arial", 10, "bold"), bg="#f4f6f9", fg="#d35400").pack(anchor="w", pady=(5,2))

colunas_tabela = ("vizinho", "ch", "mat", "lc", "cn", "red", "status")
tabela_vizinhos = ttk.Treeview(frame_res, columns=colunas_tabela, show="headings", height=5)
tabela_vizinhos.heading("vizinho", text="Instância")
tabela_vizinhos.heading("ch", text="Nota CH")
tabela_vizinhos.heading("mat", text="Nota MAT")
tabela_vizinhos.heading("lc", text="Nota LC")
tabela_vizinhos.heading("cn", text="Nota CN")
tabela_vizinhos.heading("red", text="Nota RED")
tabela_vizinhos.heading("status", text="Resultado Final")

tabela_vizinhos.column("vizinho", width=80, anchor="center")
tabela_vizinhos.column("ch", width=80, anchor="center")
tabela_vizinhos.column("mat", width=80, anchor="center")
tabela_vizinhos.column("lc", width=80, anchor="center")
tabela_vizinhos.column("cn", width=80, anchor="center")
tabela_vizinhos.column("red", width=80, anchor="center")
tabela_vizinhos.column("status", width=125, anchor="center")
tabela_vizinhos.pack(fill="x", pady=2)

tabela_vizinhos.tag_configure("aprovado_linha", background="#D4EDDA", foreground="#155724")
tabela_vizinhos.tag_configure("reprovado_linha", background="#F8D7DA", foreground="#721C24")

tk.Label(frame_res, text="RECOMENDAÇÕES DIRECIONADAS À TOMADA DE DECISÃO:", font=("Arial", 10, "bold"), bg="#f4f6f9", fg="#2c3e50").pack(anchor="w", pady=(5,0))
txt_recomendacoes = tk.Text(frame_res, height=5, font=("Arial", 10), bg="#ffffff", relief="solid", bd=1)
txt_recomendacoes.pack(fill="both", expand=True, pady=5)

janela.mainloop()