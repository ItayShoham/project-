import pygame
import sys
import copy
import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


class Connect4:
    def __init__(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.players = ['Red', 'Yellow']
        self.current_player = 0

    def is_valid_move(self, column):
        return 0 <= column < 7 and self.board[0][column] == ' '

    # def drop_piece(self, column):
    #     for row in range(5, -1, -1):
    #         if self.board[row][column] == ' ':
    #             self.board[row][column] = self.players[self.current_player]
    #             if self.current_player == 1:
    #                   self.current_player = 0
    #             else:
    #                   self.current_player = 1
    #             break

    def drop_piece(self, column):
     for row in range(5, -1, -1):
        if self.board[row][column] == ' ':
            self.board[row][column] = self.players[self.current_player]
            self.current_player = 1 - self.current_player  # This line is simplified
            break

    def undo_move(self, column):
        for row in range(6):
            if self.board[row][column] != ' ':
                self.board[row][column] = ' '
                self.current_player = 1 - self.current_player
                break

    def check_win(self):
        for row in range(6):
            for col in range(7):
                if self.board[row][col] != ' ':
                    if col + 3 < 7 and self.board[row][col] == self.board[row][col + 1] == \
                            self.board[row][col + 2] == self.board[row][col + 3]:
                        return True
                    if row + 3 < 6 and self.board[row][col] == self.board[row + 1][col] == \
                            self.board[row + 2][col] == self.board[row + 3][col]:
                        return True
                    if col + 3 < 7 and row + 3 < 6 and self.board[row][col] == self.board[row + 1][col + 1] == \
                            self.board[row + 2][col + 2] == self.board[row + 3][col + 3]:
                        return True
                    if col - 3 >= 0 and row + 3 < 6 and self.board[row][col] == self.board[row + 1][col - 1] == \
                            self.board[row + 2][col - 2] == self.board[row + 3][col - 3]:
                        return True
        return False

    def is_tie(self):
        return all(self.board[i][j] != ' ' for i in range(6) for j in range(7))

    def evaluate_window(self, window):
        score = 0
        opponent = 'Red' if self.current_player == 1 else 'Yellow'
        if window.count(self.players[self.current_player]) == 4:
            score += 100
        elif window.count(self.players[self.current_player]) == 3 and window.count(' ') == 1:
            score += 5
        elif window.count(self.players[self.current_player]) == 2 and window.count(' ') == 2:
            score += 2
        if window.count(opponent) == 3 and window.count(' ') == 1:
            score -= 4
        return score

    def score_position(self):
        score = 0
        center_array = [self.board[i][3] for i in range(6)]
        center_count = center_array.count(self.players[self.current_player])
        score += center_count * 3
        for row in range(6):
            row_array = self.board[row]
            for col in range(7-3):
                window = row_array[col:col+4]
                score += self.evaluate_window(window)
        for col in range(7):
            col_array = [self.board[row][col] for row in range(6)]
            for row in range(6-3):
                window = col_array[row:row+4]
                score += self.evaluate_window(window)
        for row in range(6-3):
            for col in range(7-3):
                window = [self.board[row+i][col+i] for i in range(4)]
                score += self.evaluate_window(window)
        for row in range(6-3):
            for col in range(7-3):
                window = [self.board[row+3-i][col+i] for i in range(4)]
                score += self.evaluate_window(window)
        return score

    def minimax(self, depth, alpha, beta, maximizing_player):
        valid_locations = [col for col in range(7) if self.is_valid_move(col)]
        is_terminal = self.check_win() or self.is_tie()
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_win():
                    return (None, 100000000000000 if maximizing_player else -10000000000000)
                else:
                    return (None, 0)
            else:
                return (None, self.score_position())
        if maximizing_player:
            value = -np.inf
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                temp_player = self.current_player
                self.drop_piece(col)
                new_score = self.minimax(depth-1, alpha, beta, False)[1]
                self.undo_move(col)
                self.current_player = temp_player
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value
        else:
            value = np.inf
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                temp_player = self.current_player
                self.drop_piece(col)
                new_score = self.minimax(depth-1, alpha, beta, True)[1]
                self.undo_move(col)
                self.current_player = temp_player
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def draw_board(self, screen):
        hole_size = 87.3
        circle_radius = 30  # Size of each circle
        horizontal_offset = 15
        vertical_offset = 15.75

        for x in range(7):
            for y in range(6):
                circle_center_x = x * hole_size + hole_size // 2 + horizontal_offset
                circle_center_y = y * hole_size + hole_size // 2 + vertical_offset

                if self.board[y][x] == 'Red':
                    pygame.draw.circle(screen, (255, 0, 0), (circle_center_x, circle_center_y), circle_radius)
                elif self.board[y][x] == 'Yellow':
                    pygame.draw.circle(screen, (255, 255, 0), (circle_center_x, circle_center_y), circle_radius)


class Connect4Dataset:
    def __init__(self):
        self.data = []

    def generate_game_states(self, game, depth=0, max_depth=3):
        if depth == max_depth:
            return
        for col in range(7):
            if game.is_valid_move(col):
                cloned_game = copy.deepcopy(game)
                cloned_game.drop_piece(col)
                encoded_state = self.encode_game_state(cloned_game)
                label = self.get_label(cloned_game)
                self.data.append((encoded_state, label))
                self.generate_game_states(cloned_game, depth + 1, max_depth)

    def encode_game_state(self, game):
        encoded_state = []
        for row in game.board:
            for cell in row:
                if cell == ' ':
                    encoded_state.extend([1, 0, 0])
                elif cell == 'Yellow':
                    encoded_state.extend([0, 1, 0])
                elif cell == 'Red':
                    encoded_state.extend([0, 0, 1])
        return encoded_state

    def get_label(self, game):
        label = [0] * 7
        label[0] = 1  # Placeholder for the optimal move
        return label




class Game:
    def __init__(self):
        pygame.display.init()  # Initialize only the display module
        pygame.font.init()  # Initialize the font module

        self.screen_width = 639
        self.screen_height = 553
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Connect Four")

        self.background_image = pygame.image.load("Connect4Board.png").convert()
        self.background_image = pygame.transform.scale(self.background_image, (self.screen_width, self.screen_height))

        self.clock = pygame.time.Clock()
        self.connect4_game = Connect4()
        self.game_over = False
        self.winner = None
        self.game_over_time = None

        self.play_again_button = pygame.Rect(0, 0, 0, 0)  # Initialize play_again_button

        self.dataset_features = []
        self.dataset_labels = []

        #self.current_player = 0  # 0 for Human, 1 for AI

        # Attempt to load the trained model

        try:
            self.model = joblib.load('connect4_model.pkl')  # Adjust the filename as per your saved model
        except FileNotFoundError:
            print("Error: Model file 'connect4_model.pkl' not found.")
            sys.exit(1)  # Exit the program if the model file is not found

    # def handle_events(self):
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             sys.exit()

    #         if not self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
    #             if 0 <= event.pos[0] <= self.screen_width:
    #                 column = event.pos[0] // 91
    #                 if self.connect4_game.is_valid_move(column):
    #                     current_player_before_move = self.connect4_game.current_player
    #                     self.connect4_game.drop_piece(column)
    #                     if self.connect4_game.check_win():
    #                         self.game_over = True
    #                         self.winner = self.connect4_game.players[current_player_before_move]
    #                         self.game_over_time = pygame.time.get_ticks()
    #                     elif self.connect4_game.is_tie():
    #                         self.game_over = True
    #                         self.winner = "Tie"
    #                         self.game_over_time = pygame.time.get_ticks()
    #                     else:
    #                         self.current_player = (self.current_player + 1) % 2
    #                         if self.current_player == 1:  # AI's turn
    #                             self.get_ai_move()

    #         if self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
    #             if self.play_again_button.collidepoint(event.pos):
    #                 self.reset_game()
    #                 self.game_over = False
    #                 self.winner = None
    #                 self.game_over_time = None

    def handle_events(self):
     for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if not self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
            if 0 <= event.pos[0] <= self.screen_width:
                column = event.pos[0] // 91
                if self.connect4_game.is_valid_move(column):
                    current_player_before_move = self.connect4_game.current_player
                    self.connect4_game.drop_piece(column)
                    if self.connect4_game.check_win():
                        self.game_over = True
                        self.winner = self.connect4_game.players[current_player_before_move]
                        self.game_over_time = pygame.time.get_ticks()
                    elif self.connect4_game.is_tie():
                        self.game_over = True
                        self.winner = "Tie"
                        self.game_over_time = pygame.time.get_ticks()
                    else:
                        self.get_ai_move()  # AI's turn after player's move

        if self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
            if self.play_again_button.collidepoint(event.pos):
                self.reset_game()
                self.game_over = False
                self.winner = None
                self.game_over_time = None

    # def get_ai_move(self):
    #     col, minimax_score = self.connect4_game.minimax(5, -np.inf, np.inf, True)
    #     if self.connect4_game.is_valid_move(col):
    #         current_player_before_move = self.connect4_game.current_player
    #         self.connect4_game.drop_piece(col)
    #         if self.connect4_game.check_win():
    #             self.game_over = True
    #             self.winner = self.connect4_game.players[current_player_before_move]
    #             self.game_over_time = pygame.time.get_ticks()
    #         elif self.connect4_game.is_tie():
    #             self.game_over = True
    #             self.winner = "Tie"
    #             self.game_over_time = pygame.time.get_ticks()
    #         else:
    #             self.current_player = (self.current_player + 1) % 2

    def get_ai_move(self):
        col, minimax_score = self.connect4_game.minimax(5, -np.inf, np.inf, True)
        if self.connect4_game.is_valid_move(col):
            current_player_before_move = self.connect4_game.current_player
            self.connect4_game.drop_piece(col)
            if self.connect4_game.check_win():
                self.game_over = True
                self.winner = self.connect4_game.players[current_player_before_move]
                self.game_over_time = pygame.time.get_ticks()
            elif self.connect4_game.is_tie():
                self.game_over = True
                self.winner = "Tie"
                self.game_over_time = pygame.time.get_ticks()

    def encode_board(self, board):
        encoded_state = []
        for row in board:
            for cell in row:
                if cell == ' ':
                    encoded_state.extend([1, 0, 0])
                elif cell == 'Yellow':
                    encoded_state.extend([0, 1, 0])
                elif cell == 'Red':
                    encoded_state.extend([0, 0, 1])
        return np.array(encoded_state)

    def draw(self):
        self.screen.blit(self.background_image, (0, 0))
        self.connect4_game.draw_board(self.screen)
        if self.game_over:
            self.draw_winner_screen()
        pygame.display.update()

    def draw_winner_screen(self):
        custom_font_path = "CustomFont2.ttf"
        font_size = 25
        custom_font = pygame.font.Font(custom_font_path, font_size)
        if self.winner == "Tie":
            text = custom_font.render("It's a tie!", True, (0, 0, 0))
        else:
            text = custom_font.render(f"The {self.winner} Player Has Won! Congrats!", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text, text_rect)

        if self.game_over_time is not None and pygame.time.get_ticks() - self.game_over_time >= 5000:
            self.draw_play_again_button()

    def draw_play_again_button(self):
        custom_font_path = "CustomFont2.ttf"
        font_size = 25
        custom_font = pygame.font.Font(custom_font_path, font_size)
        play_again_text = custom_font.render("Play Again", True, (255, 255, 255))
        play_again_rect = play_again_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 85))
        self.play_again_button = play_again_rect
        self.screen.blit(play_again_text, play_again_rect)

    def reset_game(self):
        self.connect4_game = Connect4()
        self.dataset_features = []
        self.dataset_labels = []

    def run(self):
        quit_game = False
        while not quit_game:
            self.handle_events()
            self.draw()
            self.clock.tick(30)
        pygame.quit()

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
