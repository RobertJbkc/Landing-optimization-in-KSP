# Landing-optimization-in-KSP
Kerbal Space Program (KSP) é um jogo/simulador espacial onde é possível explorar o sistema solar de Kerbol sob regime da Física realista. O principal foco do jogo é o lançamento de foguetes. Tal qual nas explorações espaciais, o objetivo é explorar o máximo possível outros planetas e coletar ciência (fazer pesquisa).

Há dois cenários onde o pouso de foguete é facilmente imaginável. Quando tem-se por objetivo economizar recursos para o lançamento de um próximo foguete, como o que já é realizado por empresas como a SpaceX e o programa espacial chinês, ou quando se quer pousar em outro corpo celeste, seja para retornar depois, ou para uma descida que não pode ser realizada com paraquedas e amortecedores de impacto, como os Rovers Curiosity e Perseverance, que usma de um sistema de descida assistido de foguetes chamado *sky crane* (guindaste aéreo).

Nesses casos, por mais que existam sistemas analíticos para modelar o problema de pouso, mudança nas condições atmosféricas ou funcionais do foguete, como a perda de um motor, ou mesmo desconhecimento das condições climáticas exatas do horário de pouso - seja na Terra ou em outro planeta -- podem levar à não realização do pouso.

Assim, o objetivo do trabalho é desenvolver um algoritmo de aprendizado por reforço profundo (DRL) que aprenda como realizar o pouso de um foguete de forma segura e com economia de combustível. Em outras palavras, espera-se que o algoritmo seja capaz de aprender a política para a manobra de Hoverslam, onde os motores são completamente ativados no último instante para que a velocidade com que a nave toque o solo seja nula.

### Termos importantes
No aprendizado por reforço, seja profundo (com redes neurais) ou não, é importante definir os termos de conversa para que seja possível estabelecer uma boa comunicação.

O **agente** é quem controla o foguete. Este toma **ações** segundo uma **política** representada por $\pi(a|s)$, que representa a probabilidade de tomar a ação $a$ dado que está em um **estado** $s$. O estado é formado por observações do ambiente, como velocidade vertical, combustível, altitude... A cada ação tomada há uma **recompensa** associada. O objetivo do algoritmo é, então, maximizar essa recompensa.

Os algoritmos trabalham tentando aproximar uma **função de valor** $V(s)$ ou a própria política. Valor é uma função que descreve o **retorno esperado** futuro ao executar uma ação em um dado estado.

## O trabalho
Neste trabalho foi implementado um algoritmo do tipo Actor-Critic, caracterizado por juntar tanto a aproximação da função de valor quanto a política. Este é o Proximal Policy Optimization (PPO), que tem o objetivo, como o nome indica, de otimizar a política de forma proximal, ou seja, dando passos menores, que levam a um aprendizado mais suave.

 Assim, há uma rede neural de duas cabeças onde cada cabeça aproxima uma das funções. **Actor** é responsável por aprender a prever os parâmetros da distribuição da qual a ação é amostrada, e **Critic** deve aprender o retorno esperado. Baseando-se na estimativa da função de valor e nas recompensas adquiridas, calcula-se a **vantagem**
$$A(s, a) = Q(s, a) - V(s),$$
que mede o quanto uma ação produziu um retorno
superior ou inferior ao esperado para aquele es-
tado. Valores positivos indicam que a ação foi
melhor do que o previsto pelo Critic, enquanto
valores negativos indicam um desempenho infe-
rior ao esperado. Essa quantidade é utilizada pelo
PPO para direcionar a atualização da política, re-
forçando ações vantajosas e reduzindo a probabi-
lidade de ações desfavoráveis.

Para mais detalhes da implementação matemática, consultar a seção []

## Organização
O trabalho é organizado em 5 scripts, sendo esses descritos na tabela abaixo.
<table>
  <thead>
    <tr>
      <th>Arquivo</th>
      <th>Descrição</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>train.py</code></td>
      <td>Coordena todos os outros arquivos, possuí o loop principal que conta o número de episódios que já se decorreram. </td>
    </tr>
    <tr>
      <td><code>ksp_environment.py</code></td>
      <td>Faz a intermediação do programa com o jogo por meio do módulo Kerbal Remote Procedure Call (KRPC). Também é responsável por calcular a recompensa por passo dado característica como velocidade e altitude.</td>
    </tr>
    <tr>
      <td><code>rollout_buffer.py</code></td>
      <td>É uma memória dos estados do sistema. Possui um limite de espaço. Após 𝑛 observações train.py chama a atualização e apaga a memória.</td>
    </tr>
    <tr>
      <td><code>actor_critic.py</code></td>
      <td>Tem a função de criar a rede neural multicamadas (MLP) de duas cabeças de forma paramétrica.</td>
    </tr>
    <tr>
      <td><code>ppo.py</code></td>
      <td>É o algoritmo de DRL que atualiza os pesos da rede. Contém o GAE e as losses de cada rede.</td>
    </tr>
  </tbody>
</table>

#### Termos para entendimento
Assim como em redes neurais o loop principal é função das épocas, no caso do aprendizado por reforço, são episódios. O algoritmo de DRL deve guiar o aprendizado, nesse caso atualizando pesos de uma rede neural.

GAE é o **Generalized Advantage Estimation** é uma consequência de não conhecer com exatidão quem é a função $Q$ para o cálculo das vantagens. Isso leva a uma aproximação de alta variância e pouco viés. A alta variância é corrigida ou amenizada via o GAE, desenvolvido por Schulman et al.