import krpc
import torch
from time import sleep

from rocket_lander.memory.rollout_buffer import RolloutBuffer


class KSPEnvironment():

    def __init__(self, frequencia: float) -> None:

        self.dt = 1 / frequencia

        self.conn = krpc.connect('Rocket Lander') # Comunicação com o KSP
        self.space_center = self.conn.space_center # Interface principal do Jogo
        self.nave = self.space_center.active_vessel # Nave que está sendo controlada
        self.control = self.nave.control # Comandos relativos à nave
        self.flight = self.nave.flight(self.nave.orbit.body.reference_frame) # Sensores da nave

        self.nome_save = 'Start'

        self.inicio = True


    def reset(self) -> torch.Tensor:
        """Restaura o ambiente a uma situação original padronizada. Fururamente pode aplicar pequnas variações para generalizar o treinamento"""

        self._reset_nave()
        self.inicio = True # Permite atualizar as variáveis de início
        self.control.sas = True
        self.control.sas_mode = self.space_center.SASMode.retrograde

        return self._get_state()

    def step(self, action: torch.Tensor, buffer: RolloutBuffer) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Dá um passo na simulacão. Como o simulador é um jogo, não há um passo de simulação, mas sim uma espera por um tempo que determina uma frequência de leietura.

        Args:
            action (torch.Tensor): A ação que deve ser tomada.

        Returns:
            tuple[torch.Tensor, float, bool]: O novo estado adquirido, o quão bom foi realizar a ação (recompensa) e se o episódio terminou ou não.
        """

        self._envia_acao(action=action)
        sleep(self.dt)
        novo_estado = self._get_state()
        recompensa = torch.tensor(self._calc_recompensa(novo_estado, buffer))
        done, _ = self._is_done()

        return novo_estado, recompensa, done


    def _get_state(self) -> torch.Tensor:
        """Lê o estado atual da nave e o converte para um tensor. O estado engloba ambiente e "sensores" da nave.

        Aplica uma normalização baseada no valor máximo de cada atributo.

        Returns:
            torch.Tensor: Um tensor representando o estado do sistema.
        """

        self.flight = self.nave.flight(self.nave.orbit.body.reference_frame) # "Sensores" da nave
        if self.inicio:
            self.altitude_i = self.flight.surface_altitude
            self.velocidade_vertical_i = abs(self.flight.vertical_speed)
            self.velocidade_horizontal_i = abs(self.flight.horizontal_speed)
            self.velocidade_i = [abs(i) for i in self.flight.velocity]
            self.propelente_i = self.nave.resources.amount('LiquidFuel')
            self.gravidade_i = abs(self.nave.orbit.body.surface_gravity)
            self.massa_i = self.nave.mass
            self.thrust_i = self.nave.max_thrust # Ou max?
            self.weight_i = self.nave.mass * abs(self.nave.orbit.body.surface_gravity)
            # Cálculo do TWR - Thrust to Weight Ratio
            self.twr_i = self.nave.max_thrust / (self.nave.mass * abs(self.nave.orbit.body.surface_gravity)) if self.nave.mass > 0 else 0
            # Talvez trocar para available_thrust ou max_thrust
            ## Explodir é gastar todo o combistível!...
            
            self.inicio = False # Faz com que os valores iniciais seja guardados

        # ===== Os volores abaixo estão normalizados pelo valor inicial
        altitude = self.flight.surface_altitude / self.altitude_i
        velocidade_vertical = self.flight.vertical_speed / self.velocidade_vertical_i
        velocidade_horizontal = self.flight.horizontal_speed / self.velocidade_horizontal_i
        propelente = self.nave.resources.amount('LiquidFuel') / self.propelente_i
        gravidade = abs(self.nave.orbit.body.surface_gravity) / self.gravidade_i
        # Cálculo do TWR - Thrust to Weight Ratio
        massa_twr = self.nave.mass
        twr_max = (self.nave.max_thrust / (massa_twr * abs(self.nave.orbit.body.surface_gravity))) / self.twr_i if massa_twr > 0 else 0

        estado = (
            altitude,
            velocidade_vertical,
            velocidade_horizontal,
            propelente,
            gravidade,
            twr_max
        )

        # print('Estado antes de tensor:', estado)

        return torch.tensor(estado, dtype=torch.float32)

    def _calc_recompensa(self, estado, buffer: RolloutBuffer):
        """Definie como é calculada a recompensa. Faz com que os seguintes obeetivos sejam priorizados:
        - O foguete deve atingir o solo
        - A velocidade vertical deve ser a mais próxima de 0 possível
        - Sem EXPLODIR (penalizar esse caso). Acabar o combustível antes do pouso é o mesmo que EXPLODIR
        - Gastar o mínimo de combustível

        O agente deve andar na direção de aumentar (maximizar) a recompensa.
        """

        self.flight = self.nave.flight(self.nave.orbit.body.reference_frame)
        altitude = self.flight.surface_altitude
        velocidade = self.flight.velocity # Um vetor
        massa = self.nave.mass
        gravity = self.nave.orbit.body.surface_gravity
        thrust = self.nave.thrust
        weight = massa * gravity
        # Cálculo do TWR - Thrust to Weight Ratio
        twr = thrust / weight if weight > 0 else 0

        # Para mostrar como é o estado:
        # estado[0] = altitude
        # estado[1] = velocidade_vertical
        # estado[2] = velocidade_horizontal
        # estado[3] = propelente
        # estado[4] = gravidade
        # estado[5] = twr
        

        # k = (estado[0]) + 0.01
        # vx = abs(velocidade[0]) / self.velocidade_i[0]
        # vy = abs(velocidade[1]) / self.velocidade_i[1]
        # vz = abs(velocidade[2]) / self.velocidade_i[2]

        # penalidade_movimento = - (1/100) * ((vx / k) + (vy / k) + (vz / k))

        # penalidade_movimento_2 = - 1 * abs(estado[2]) - 1 * abs(estado[1])

        # altura_passada = buffer.buffer['estados'][0][-1] if buffer.size > 0 else altitude # Ou -1 0
        # progresso = (1/10) * (altura_passada - estado[0])


        # print(f'Rec:: Pen mov: {penalidade_movimento}, Pen mov 2: {penalidade_movimento_2}, prog: {progresso}')
        # recompensa = 0
        # recompensa += penalidade_movimento + penalidade_movimento + progresso


        w1, w2, w3 = 1, 3, 0.4
        altura_passada = buffer.buffer['estados'][-1][0] if buffer.size > 0 else altitude / self.altitude_i # Ou -1 0
        penalidade_movimento = - w1 * abs(estado[2]) - w2 * abs(estado[1]) + w3 * (altura_passada - estado[0])

        # print(f'BufBuf: {buffer.buffer['estados'][-1][0] if buffer.size > 0 else altitude / self.altitude_i}, Est2: {estado[2]}, Est1: {estado[1]}')

        recompensa = 0
        recompensa += penalidade_movimento



        self.bonus_pouso = 20
        self.penalidade = -20

        # Com o método de recompensas variáveis devo verificar se foi combustível ou não
        situacao = self._is_done()
        if situacao[0]:
            if situacao[1]:
                recompensa += self.bonus_pouso
            else:
                recompensa += self.penalidade * (abs(estado[1]))
                print(f'Penalidade: {self.penalidade * (abs(estado[1]))}')

        return recompensa

    def _envia_acao(self, action: torch.Tensor) -> None:
        """Envia as ações para o jogo"""

        self.control.throttle = float(action[0])

    def _is_done(self) -> tuple[bool, bool]:
        """Verifica se o episódio terminou.

        Returns:
            bool: True caso o episódio tenha terminado.
        """

        self.flight = self.nave.flight(self.nave.orbit.body.reference_frame) # "Sensores" da nave    
        altitude = self.flight.surface_altitude
        velocidade_vertical = self.flight.vertical_speed

        if abs(velocidade_vertical) < 3 and altitude < 10:
            return True, True

        if abs(velocidade_vertical) > 20 and altitude < 5:
            return True, False
        
        if self.nave.situation == 'landed':
            return True, True

        if self.nave.situation == 'splashed':
            return True, False

        if self.nave.resources.amount('LiquidFuel') <= 0:
            return True, False

        return False, False

    def _reset_nave(self) -> None:
        """Reseta a posição e condições originais da nave. A base de funiconamento é o load de um save com a nave em posição. """

        self.space_center.load(self.nome_save)

        # Copiados para garantir que a referências estão corretas após o load do save.
        self.space_center = self.conn.space_center # Interface principal do Jogo
        self.nave = self.space_center.active_vessel # type: ignore # Nave que se controla
        self.control = self.nave.control # Comandos relativos à nave
        self.flight = self.nave.flight(self.nave.orbit.body.reference_frame) # Sensores da nave