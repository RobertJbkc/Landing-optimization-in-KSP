import krpc
import torch
from time import sleep


class KSPEnvironment():

    def __init__(self, frequencia: float):

        self.dt = 1 / frequencia

        self.conn = krpc.connect('Rocket Lander') # Comunicação com o KSP
        self.space_center = self.conn.space_center # Interface principal do Jogo
        self.nave = self.space_center.active_vessel # type: ignore # Nave que se controla
        self.control = self.nave.control # Comandos relativos à nave

        self.refframe = self.nave.surface_reference_frame # Referência espacial (sistema de coordenadas)
        self.flight = self.nave.flight(reference_frame=self.refframe) # "Sensores" da nave

        # constantes para o blanceameto da loss
        self.w_altitude = 1
        self.w_vertical_speed = 1
        self.w_horizontal_speed = 0
        self.w_fuel = -1/10
        self.bonus_pouso = 100
        self.penalidade_explosao = -100

        self.nome_save = 'Start'


    def reset(self):
        """Deve resetar o ambiente para o treino do modelo. Aplicar uma variacão aleatória."""

        self._reset_nave()
        # self.nave.auto_pilot.disengage()
        self.control.sas = True
        self.control.sas_mode = self.space_center.SASMode.retrograde

        return self._get_state()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        """Dá um passo na simulacão. Como o simulador é um jogo, não há um passo de simulação, mas sim uma espera por um tempo que determina uma frequência de leietura.

        Args:
            action (torch.Tensor): A ação que deve ser tomada.

        Returns:
            tuple[torch.Tensor, float, bool]: O novo estado adquirido, o quão bom foi realizar a ação (recompensa) e se o episódio terminou ou não.
        """

        self._envia_acao(action=action)
        sleep(self.dt)
        novo_estado = self._get_state()
        recompensa = self._calc_recompensa(novo_estado)
        done, _ = self._is_done()

        return novo_estado, recompensa, done


    def _get_state(self) -> torch.Tensor:
        """Lê o estado atual da nave a converte para um tensor.

        Returns:
            torch.Tensor: Um tensor representando o estado do sistema.
        """

        # self.refframe = self.nave.surface_reference_frame # Referência espacial (sistema de coordenadas)
        # self.flight = self.nave.flight(reference_frame=self.refframe) # "Sensores" da nave
        self.flight = self.nave.flight(self.nave.orbit.body.reference_frame) # "Sensores" da nave
        

        # Ler "sensores"
        altitude = self.flight.bedrock_altitude # Ou surface
        velocidade_vertical = self.flight.vertical_speed
        velocidade_horizontal = self.flight.horizontal_speed
        massa = self.nave.mass
        propelente = self.nave.resources.amount('LiquidFuel') # Supomos, pelo amor de Deus, que o pouso seja com um motor a combustível líquido

        # Construir o estado
        estado = (
            altitude,
            velocidade_vertical,
            velocidade_horizontal,
            massa,
            propelente
        )

        # Converter para tensor
        return torch.tensor(estado, dtype=torch.float32)

    def _calc_recompensa(self, estado):
        """Objetivos de aprendizado:
        - O foguete deve atingir o solo
        - A velocidade vertical deve ser a mais próxima de 0 possível
        - A velocidade horizontal deve ser próxima de zero tal qual a vertical (principalmente em um caso completo)
        - Gastando o mínimo de combustível
        - Sem EXPLODIR (penalizar esse caso)

        Uma função será minimizada, esta é a função de perda. Ela deve ter duas partes: a contínua cuida dos eventos contínuos durante o voo e a discreta está relacionada a uma nota para o voo. Esta última podendo ser negativa.
        """
        
        perda = 0
        perda -= self.w_altitude * abs(estado[0])
        # perda -= self.w_vertical_speed * abs(estado[1])
        perda -= self.w_vertical_speed * estado[1] # Tem que ser assim
        # Adicionar tratamento especial se subir
        perda -= self.w_horizontal_speed * abs(estado[2])
        perda -= self.w_fuel * abs(estado[4])

        situacao = self._is_done()
        if situacao[0]: # Se pousou
            if situacao[1]: # Se foi de uma bom modo
                return perda * self.bonus_pouso
            else:
                return perda * self.penalidade_explosao
        
        return perda

    def _envia_acao(self, action: torch.Tensor):
        """Envia as ações para o jogo"""

        ### O método de construir a MLP deve ser alterado para que a camada de saída seja definida com uma possível funcão diferente.
        # Quem deve garantir que o throttle pertence ao intervalo [0, 1] é a rede neural com uma finalização sigmoide
        self.control.throttle = float(action[0])

    def _is_done(self) -> tuple[bool, bool]:
        """Verifica se o episódio terminou.

        Returns:
            bool: True caso o episódio tenha terminado.
        """
        self.flight = self.nave.flight(self.nave.orbit.body.reference_frame) # "Sensores" da nave
                
        
        # Ler "sensores"
        altitude = self.flight.bedrock_altitude # Ou surface
        velocidade_vertical = self.flight.vertical_speed

        if velocidade_vertical < 1 and altitude < 5:
            return True, True
        
        if self.nave.situation == 'landed':
            return True, True

        if self.nave.situation == 'splashed':
            return True, False

        if self.nave.resources.amount('LiquidFuel') <= 0:
            return True, False

        return False, False # O segundo não deve ser usado. É um preenchimento.

    def _reset_nave(self):
        """Reseta a posição e condições originais da nave. Será implementada uma variação aleatória nosparâmetros. A base de funiconamento é o load de um save com a nave em posição. """
        self.space_center.load(self.nome_save) # type: ignore

        # Copiados para garantir que a referências estão corretas após o load do save.
        self.space_center = self.conn.space_center # Interface principal do Jogo
        self.nave = self.space_center.active_vessel # type: ignore # Nave que se controla
        self.control = self.nave.control # Comandos relativos à nave

        self.refframe = self.nave.surface_reference_frame # Referência espacial (sistema de coordenadas)
        self.flight = self.nave.flight(reference_frame=self.refframe) # "Sensores" da nave