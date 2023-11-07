from Deck import Card


class Player:
    def __init__(self):
        self.hand = []
        self.ace_count = 0

    def sum_hand(self):
        hand_sum = 0
        for card in self.hand:
            hand_sum += int(card.value)

        ace_count = self.ace_count
        while hand_sum <= 11 and ace_count > 0:
            hand_sum += 10
            ace_count -= 1
        return hand_sum

    def draw_card(self, deck):
        card = deck.draw_card()
        if card == Card.ACE:
            self.ace_count += 1
        self.hand.append(card)
        return card
