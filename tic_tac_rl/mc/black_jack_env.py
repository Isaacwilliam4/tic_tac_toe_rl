import numpy as np
from typing import Tuple, List, Optional


def create_deck() -> List[int]:
    # Cards 1 to 9: 4 each, Card 10: 16 (10, J, Q, K)
    deck = [card for card in range(1, 10) for _ in range(4)]
    deck += [10] * 16
    np.random.shuffle(deck)
    return deck


def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21


def hand_value(hand):
    val = sum(hand)
    if usable_ace(hand):
        return val + 10
    return val


def is_bust(hand):
    return hand_value(hand) > 21


class Blackjack:
    def __init__(self):
        self.deck: List[int] = []
        self.player_hand: List[int] = [0 for _ in range(6)]
        self.dealer_hand: List[int] = [0]
        self.done: bool = False

    def draw_card(self) -> int:
        if len(self.deck) == 0:
            raise ValueError("The deck is empty")
        return self.deck.pop()

    def draw_hand(self) -> List[int]:
        return [self.draw_card(), self.draw_card()]

    def reset(self) -> Tuple[Tuple[int, bool, int], int]:
        self.deck = create_deck()
        self.player_hand = self.draw_hand()
        self.dealer_hand = self.draw_hand()
        self.done = False
        return self.get_state(), 0

    def get_state(self) -> Tuple[int, bool, int]:
        return (
            hand_value(self.player_hand),
            usable_ace(self.player_hand),
            self.dealer_hand[0]
        )

    def get_available_actions(self) -> List[int]:
        return [0, 1] if not self.done else []

    def is_terminal(self) -> bool:
        return self.done

    def step(self, action: int) -> Tuple[Tuple[int, bool, int], int, bool]:
        if self.done:
            raise ValueError("Game is already over")

        if action == 1:  # hit
            self.player_hand.append(self.draw_card())
            if is_bust(self.player_hand):
                self.done = True
                return self.get_state(), -1, self.done
            else:
                return self.get_state(), 0, self.done

        elif action == 0:  # stick
            while hand_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self.draw_card())

            player_val = hand_value(self.player_hand)
            dealer_val = hand_value(self.dealer_hand)
            self.done = True

            if is_bust(self.dealer_hand) or player_val > dealer_val:
                reward = 1
            elif player_val == dealer_val:
                reward = 0
            else:
                reward = -1

            return self.get_state(), reward, self.done

        else:
            raise ValueError("Invalid action. Use 0 (stick) or 1 (hit)")

    def print_state(self):
        print(f"Player hand: {self.player_hand} | value: {hand_value(self.player_hand)} | usable ace: {usable_ace(self.player_hand)}")
        print(f"Dealer showing: {self.dealer_hand[0]}")
        if self.done:
            print(f"Dealer full hand: {self.dealer_hand} | value: {hand_value(self.dealer_hand)}")
        print()
