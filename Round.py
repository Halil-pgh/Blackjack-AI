import os
from Deck import Deck
from Player import Player
from AI import *


class Round:
    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2

    def start(self):
        state_space = 3  # sum, ace count, other player's open card
        action_space = 2  # "hit" or "stand"
        learning_rate = 0.001
        discount_factor = 0.99
        epsilon = 0.1
        agent = QLearningAgent(state_space, action_space, learning_rate, discount_factor, epsilon)

        old_player1_win_count = 0
        old_player2_win_count = 0
        player1_win_count = 0
        player2_win_count = 0
        while True:
            deck = Deck()
            deck.shuffle()
            self.player1 = Player()
            self.player2 = Player()

            self.player2.draw_card(deck)
            self.player2.draw_card(deck)
            self.player1.draw_card(deck)
            self.player1.draw_card(deck)
            # print("Dealer's hand: ? -", self.player2.draw_card(deck))
            # print("AI's hand:", self.player1.draw_card(deck), "-", self.player1.draw_card(deck))

            already_lost = False
            state = np.array(self.get_state(), dtype=np.float32)
            next_state = state
            reward = 0
            while True:
                action = agent.select_action(state)
                if action == 0:
                    self.player1.draw_card(deck)
                    # print("AI:", self.player1.draw_card(deck))
                else:
                    break
                next_state = np.array(self.get_state(), dtype=np.float32)
                if self.player1.sum_hand() > 21:
                    already_lost = True
                    break
                agent.update_q_values(state, action, reward, next_state)
                state = next_state

            # print("AI sum:", self.player1.sum_hand())

            player1_win = False
            if not already_lost:
                while self.player2.sum_hand() <= self.player1.sum_hand():
                    self.player2.draw_card(deck)
                    # print("Dealer:", self.player2.draw_card(deck))
                    if self.player2.sum_hand() > 21:
                        player1_win = True
                        break
                # print("Dealer sum:", self.player2.sum_hand())

            if player1_win:
                player1_win_count += 1
                reward = 10
            else:
                player2_win_count += 1
                reward = -10
            agent.update_q_values(state, action, reward, next_state)

            os.system("cls")
            print(f"AI' win count: {player1_win_count}")
            print(f"Dealer's win count: {player2_win_count}")
            print(tf.config.list_physical_devices('GPU'))
            print(tf.test.is_built_with_cuda())
            if (player1_win_count + player2_win_count) % 100 == 0:
                player1_win_diff = player1_win_count - old_player1_win_count
                player2_win_diff = player2_win_count - old_player2_win_count
                file = open("ai_race_less_layers.txt", "a")
                file.write(f"AI: {player1_win_count}, Dealer: {player2_win_count}\n")
                file.write(f"Total win rate: {(player1_win_count / (player1_win_count + player2_win_count) * 100):.2f}%\n")
                file.write(f"Last 100 match win rate {(player1_win_diff / (player1_win_diff + player2_win_diff) * 100):.2f}%\n")
                file.write(f"AI's hand: {self.player1.hand.__str__()}\n")
                file.write(f"Dealer's hand: {self.player2.hand.__str__()}\n")
                file.close()
                old_player1_win_count = player1_win_count
                old_player2_win_count = player2_win_count

                if epsilon > 0.001:
                    epsilon -= 0.00001

    def get_state(self):
        return [self.player1.sum_hand(), self.player1.ace_count, self.player2.hand[1].value]
