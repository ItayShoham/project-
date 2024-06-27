import h5py
import numpy as np
import os

class Connect4:
    def __init__(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.players = ['Red', 'Yellow']
        self.current_player = 0
        self.game_over = False

    def is_valid_move(self, column):
        return 0 <= column < 7 and self.board[0][column] == ' '

    def drop_piece(self, column):
        for row in range(5, -1, -1):
            if self.board[row][column] == ' ':
                self.board[row][column] = self.players[self.current_player]
                break

    def undo_move(self, column):
        for row in range(6):
            if self.board[row][column] != ' ':
                self.board[row][column] = ' '
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

    def reset_game(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.current_player = 0
        self.game_over = False

    def make_move(self, column):
        if self.is_valid_move(column):
            self.drop_piece(column)
            if self.check_win():
                self.game_over = True
            elif self.is_tie():
                self.game_over = True
            else:
                self.current_player = (self.current_player + 1) % 2

    def get_state(self):
        state = np.zeros((3, 6, 7), dtype=int)
        for i in range(6):
            for j in range(7):
                if self.board[i][j] == 'Red':
                    state[0, i, j] = 1
                elif self.board[i][j] == 'Yellow':
                    state[1, i, j] = 1
                else:
                    state[2, i, j] = 1
        return state

    def get_valid_moves(self):
        return [col for col in range(7) if self.is_valid_move(col)]

    def minimax(self, depth, alpha, beta, maximizing_player, move_list:list = []):
        valid_moves = self.get_valid_moves()

        if depth == 0 or self.check_win() or self.is_tie():
                return move_list, self.evaluate_board()

        if maximizing_player:
            value = -float('inf')
            best_seq = move_list
            for move in valid_moves:
                self.drop_piece(move)
                new_seq , new_score = self.minimax(depth - 1, alpha, beta, False, move_list + [move])
                self.undo_move(move)
                if new_score > value:
                    value = new_score
                    best_seq = new_seq
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_seq, value
        else:
            value = float('inf')
            best_seq = move_list
            for move in valid_moves:
                self.drop_piece(move)
                new_seq, new_score = self.minimax(depth - 1, alpha, beta, True, move_list + [move])
                self.undo_move(move)
                if new_score < value:
                    value = new_score
                    best_seq = new_seq
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_seq, value

# BOARD EVAL -----------------------------------------------------------------

    # Evaluting a board score from 1000 to -1000 based on threats of each side to win. 
    

    def evaluate_board(self):
        player_score = self.evaluate_board_for_player('Red')
        opponent_score = self.evaluate_board_for_player('Yellow')
        return player_score - opponent_score

    def evaluate_board_for_player(self, player):
        score = 0
        # Directions: Horizontal, Vertical, Diagonal /, Diagonal \
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] == player:
                    for direction in directions:
                        score += self.count_potentials(self.board, row, col, direction, player)
        return score

    def count_potentials(self, board, row, col, direction, player):
        consecutive_count = 1
        for i in range(1, 4):  # Check up to three pieces in each direction
            new_row = row + direction[0] * i
            new_col = col + direction[1] * i
            if 0 <= new_row < len(board) and 0 <= new_col < len(board[0]):
                if board[new_row][new_col] == player:
                    consecutive_count += 1
                else:
                    break
            else:
                break
        # Score based on the number of pieces in line
        if consecutive_count == 2:
            return 10  # Two in a row scores 10
        elif consecutive_count == 3:
            return 100  # Three in a row scores 100
        elif consecutive_count == 4:
            return 1000  # Four in a row scores 1000 
        return 0


# BOARD EVAL -----------------------------------------------------------------

    def select_best_move(self, exploration_rate=0.2, max_depth=4):
        depth = np.random.randint(1, max_depth + 1)
        best_moves, best_value = self.minimax(depth, -float('inf'), float('inf'), True)
        if np.random.random() < exploration_rate:
            return np.random.choice(self.get_valid_moves())
        return best_moves[0]
    
    def generate_game_states(self, num_games):
        data_features = []
        data_labels = []

        for _ in range(num_games):
            self.reset_game()
            while not self.game_over:
    
                state_before_move = self.get_state()
                
                # finding the best move and puting in label and executing 
                move = self.select_best_move()
                self.make_move(move)
                
                # Record game state before move and move        
                data_features.append(state_before_move.flatten())
                data_labels.append(move)
                
                if self.check_win() or self.is_tie():
                    break

        return np.array(data_features), np.array(data_labels)

def clear_dataset(filename):

    # 'w' deletes the content and rewriting
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('features', shape=(0, 42), maxshape=(None, 42))
        hf.create_dataset('labels', shape=(0,), maxshape=(None,))

# def generate_dataset(num_games, filename):
#     connect4 = Connect4()
#     features, labels = connect4.generate_game_states(num_games)
    
#     # Ensure features are reshaped to (num_games, 3, 6, 7)
#     # features = features.reshape(num_games, 3, 6, 7)
    
#     with h5py.File(filename, 'w') as hf:
#         hf.create_dataset('features', data=features)
#         hf.create_dataset('labels', data=labels)

#     print(f"Dataset with {num_games} games generated and saved to {filename}.")

def generate_dataset(num_games, num_states_per_game, filename):
    connect4 = Connect4()
    features = []
    labels = []

    for _ in range(num_games):
        connect4.reset_game()
        game_length = np.random.randint(5, 42)
        for _ in range(game_length):
            if connect4.game_over:
                break
            state = connect4.get_state()
            move = connect4.select_best_move(exploration_rate=0.2, max_depth=4)
            features.append(state)
            labels.append(move)
            connect4.make_move(move)

    # Shuffle and cap the number of states
    combined = list(zip(features, labels))
    np.random.shuffle(combined)
    features, labels = zip(*combined[:num_states_per_game])

    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('features', data=np.array(features))
        hf.create_dataset('labels', data=np.array(labels))

    print(f"Dataset with {len(features)} states generated and saved to {filename}.")

if __name__ == "__main__":
    dataset_filename = 'connect4_large_dataset.h5'
    clear_dataset(dataset_filename)
    print("Dataset cleared successfully.")
    
    
    generate_dataset(num_games=100, num_states_per_game=10000, filename=dataset_filename)

