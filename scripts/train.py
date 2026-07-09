# Este arquivo é o grande maestro que integra dotos os outros scripts
# Deve:
# - Criar o ambiente
# - Criar a rede
# - Criar o PPO
# - Coletar experiências
# - Atualizar
# - Salvar o modelo

# PPO funciona em episódios. Até este terminar temos:
# - Observação de um estado
# - Actor escolhe uma ação
# - Ambiente executa --> retorna próximo estado, recompensa e done --> salva no buffer
# - Atualizar o PPO

import torch
import torch.nn as nn
from rocket_lander.models.actor_critic import ActorCriticNetwork
from rocket_lander.memory.rollout_buffer import RolloutBuffer
from rocket_lander.algorithms.ppo import PPO


NUM_EPISODIOS = 20
FREQUENCIA_ATUALIZACAO = 10 # Hz
NUM_PASSOS_ROLLOUT = 1000 # Ponto pegos antes de uma atualização


ambiente = Ambiente()
rede = ActorCriticNetwork(input_dim=4, camadas_ocultas=[25, 25, 25], camadas_cabecas=[10, 10], num_actions=1, ativacao=nn.Tanh)
ppo = PPO(rede, lr=5e-4, gamma=0.99, lambda_gae=0.65, epsilon_clip=0.2, coef_entropia=0.5, coef_valor=0.5, epocas=20, batch_size=32)
buffer = RolloutBuffer()



passos_coletados = 0
for i in range(NUM_EPISODIOS):
    estado = ambiente.reset() # Controla, via KRPC, o reset de tudo no jogo, assim como o novo posicionaento no mundo
    done = False

    while not done:
        acao, log_prob, valor, _ = ppo.rede.act(estado, treino=True)

        proximo_estado, recompensa, done = ambiente.step(acao)

        buffer.add(estado=estado, acao=acao, log_prob=log_prob, valor=valor, recompensa=recompensa, done=done)
        estado = proximo_estado # Fim do loop
        passos_coletados += 1

        if passos_coletados >= NUM_PASSOS_ROLLOUT:
            if done:
                proximo_valor = torch.zeros_like(valor)
            else:
                _, proximo_valor = ppo.rede(estado) # Obter a previsão do Critic
            ppo.atualizar(buffer, proximo_valor)
            passos_coletados = 0

    if buffer.size > 0: # Caso não atinja o número de passos de rollout
        ppo.atualizar(buffer, proximo_valor)
        passos_coletados = 0
