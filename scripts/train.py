"""
Arquivo destinado a conduzir o treinamento do agente de DRL via PPO. A PPO funciona em episódios.
"""

import torch
import torch.nn as nn
from rocket_lander.environment.ksp_environment import KSPEnvironment
from rocket_lander.models.actor_critic import ActorCriticNetwork
from rocket_lander.memory.rollout_buffer import RolloutBuffer
from rocket_lander.algorithms.ppo import PPO


NUM_EPISODIOS = 40
FREQUENCIA_ATUALIZACAO = 15 # Hz
NUM_PASSOS_ROLLOUT = 64 # ou 128 # Ponto pegos antes de uma atualização

ambiente = KSPEnvironment(frequencia=FREQUENCIA_ATUALIZACAO)
rede = ActorCriticNetwork(input_dim=6, camadas_ocultas=[32, 64, 64, 32], output_dim=16, camadas_cabecas=[16, 32, 16, 8], num_actions=1, ativacao=nn.Tanh, ativacao_saida=nn.Tanh)
ppo = PPO(rede, lr=3e-4, gamma=0.99, lambda_gae=0.90, epsilon_clip=0.2, coef_entropia=0.05, coef_valor=1, epocas=15, batch_size=16)
buffer = RolloutBuffer()

recompensa_para_analise = []
recompensa_por_passo = []

print('[PC] Iniciado...')
passos_coletados = 0
passos_totais = 0
for i in range(NUM_EPISODIOS):
    estado = ambiente.reset()
    done = False
    print('[PC] Reset aplicado com sucesso!')

    while not done:
        with torch.no_grad():
            acao, log_prob, valor, _ = ppo.rede.act(estado, treino=True)

        proximo_estado, recompensa, done = ambiente.step(acao, buffer=buffer)
        print(f'[info] Recompença: {recompensa}, [info] Ação: {acao}')

        buffer.add(estado=estado, acao=acao, log_prob=log_prob, valor=valor, recompensa=recompensa, done=done)
        estado = proximo_estado
        passos_coletados += 1
        passos_totais += 1

        if passos_coletados >= NUM_PASSOS_ROLLOUT:
            if done:
                proximo_valor = torch.zeros_like(valor)
            else:
                _, proximo_valor = ppo.rede(estado) # Obter a previsão do Critic
            ambiente.conn.krpc.paused = True # Pausa o jogo para atualizar as rede. Permite redes maiores
            ppo.atualizar(buffer, proximo_valor)
            ambiente.conn.krpc.paused = False
            passos_coletados = 0

    recompensa_para_analise.append((ppo.recompensa_episodio, passos_totais))
    recompensa_por_passo.append(ppo.recompensa_episodio / passos_totais)
    print(f'##### Recompensa do episódio: {ppo.recompensa_episodio:.3f}')
    print(f'Todas as recompensas: {recompensa_para_analise}')
    print(f'Recompensa por passo: {recompensa_por_passo}')
    ppo.recompensa_episodio = 0
    passos_totais = 0

    if buffer.size > 1: # Caso não atinja o número de passos de rollout. 1 para não dar problema com os desvios padrões vantagens.std(unbiased=False)
        ppo.atualizar(buffer, proximo_valor) # Esta atualização ocorre no final. Não precisa de pausa
        passos_coletados = 0
