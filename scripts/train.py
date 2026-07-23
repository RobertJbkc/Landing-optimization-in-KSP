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
from rocket_lander.environment.ksp_environment import KSPEnvironment
from rocket_lander.models.actor_critic import ActorCriticNetwork
from rocket_lander.memory.rollout_buffer import RolloutBuffer
from rocket_lander.algorithms.ppo import PPO


NUM_EPISODIOS = 20
FREQUENCIA_ATUALIZACAO = 30 # Hz
NUM_PASSOS_ROLLOUT = 50 # Ponto pegos antes de uma atualização


ambiente = KSPEnvironment(frequencia=FREQUENCIA_ATUALIZACAO)
rede = ActorCriticNetwork(input_dim=5, camadas_ocultas=[20, 20, 20], camadas_cabecas=[20, 20, 20, 1], num_actions=1, ativacao=nn.Sigmoid)
ppo = PPO(rede, lr=5e-3, gamma=0.99, lambda_gae=0.65, epsilon_clip=0.2, coef_entropia=0.1, coef_valor=0.01, epocas=20, batch_size=32)
buffer = RolloutBuffer()


print('[PC] Iniciado...')
passos_coletados = 0
for i in range(NUM_EPISODIOS):
    estado = ambiente.reset() # Controla, via KRPC, o reset de tudo no jogo, assim como o novo posicionaento no mundo
    done = False
    print('[PC] Reset aplicado com sucesso!')

    while not done:
        with torch.no_grad(): ####
            acao, log_prob, valor, _ = ppo.rede.act(estado, treino=True)
        print('Passos coletados: ', passos_coletados)

        proximo_estado, recompensa, done = ambiente.step(acao)
        print(f'[info] Recompença: {recompensa}, Done: {done}')

        buffer.add(estado=estado, acao=acao, log_prob=log_prob, valor=valor, recompensa=recompensa, done=done)
        estado = proximo_estado # Fim do loop
        passos_coletados += 1

        if passos_coletados >= NUM_PASSOS_ROLLOUT:
            if done:
                proximo_valor = torch.zeros_like(valor)
            else:
                _, proximo_valor = ppo.rede(estado) # Obter a previsão do Critic
            ppo.atualizar(buffer, proximo_valor)
            print('[AAAAA] Atualizou!!!!!')
            passos_coletados = 0

    if buffer.size > 0: # Caso não atinja o número de passos de rollout
        ppo.atualizar(buffer, proximo_valor)
        passos_coletados = 0
