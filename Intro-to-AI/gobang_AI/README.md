1.Alpha-Beta Pruning
Alpha-Beta Pruning is an optimization technique for the Minimax algorithm,
used to reduce the number of nodes evaluated in the game tree. It works by
pruning branches of the tree that cannot possibly influence the final decision,
thus speeding up the search process without affecting the result.

Alpha is the best score that the maximizing player (AI) can guarantee so
far.
Beta is the best score that the minimizing player (opponent) can guarantee
so far.
Pruning occurs when the current branch is worse than previously explored
branches, allowing the algorithm to skip evaluating that branch further.

Core Code for Alpha-Beta Pruning

def negamax(is_ai, depth, alpha, beta):
    # Check if the game is over or if search depth is reached
    if game_win(list1) or game_win(list2) or depth == 0:
        return evaluation(is_ai)

    # Get all available moves
    blank_list = list(set(list_all).difference(set(list3)))
    order(blank_list)  # Sort moves to improve pruning efficiency

    for next_step in blank_list:
        global search_count
        search_count += 1

        # Skip evaluating positions without neighboring moves
        if not has_neightnor(next_step):
            continue

        # Simulate the move
        if is_ai:
            list1.append(next_step)
        else:
            list2.append(next_step)
        list3.append(next_step)

        # Recursively call negamax and reverse the perspective
        value = -negamax(not is_ai, depth - 1, -beta, -alpha)

        # Undo the simulated move
        if is_ai:
            list1.remove(next_step)
        else:
            list2.remove(next_step)
        list3.remove(next_step)

        # Update alpha with the best value found so far
        if value > alpha:
            alpha = value
            # Store the best move at the root level
            if depth == DEPTH:
                next_point[0] = next_step[0]
                next_point[1] = next_step[1]

            # Prune branches where the maximizer's move is too good for the 
            # minimizer to handle
            if alpha >= beta:
                global cut_count
                cut_count += 1
                return beta  # Prune this branch

    return alpha
How Alpha-Beta Pruning Works:
Alpha: The maximum score the AI (maximizing player) is guaranteed.
Beta: The minimum score the opponent (minimizing player) is guaranteed.
The algorithm explores possible moves, keeping track of the best option for the
AI (maximizer) and the opponent (minimizer). If the current move guarantees a
worse outcome than what’s already explored (alpha >= beta), the branch is
pruned.

Key Concepts in the Code
Evaluation Function
The AI evaluates the game state and assigns a score based on the patterns on the
board, such as two, three, or four consecutive pieces. This function helps the
AI decide which move is most favorable.


def evaluation(is_ai):
    # Calculates the current score based on the position on the board
Search Depth
The DEPTH variable controls how far ahead the AI looks. Increasing the depth
makes the AI more strategic, but it also increases the computational cost.

Move Ordering
The order() function sorts moves to prioritize those that are more likely to
yield a better outcome, thus enhancing the effectiveness of pruning.


def order(blank_list):
    last_pt = list3[-1]
    # Prioritize moves around the last point played

2.Negamax Algorithm

Negamax is a variant of Minimax that simplifies the implementation by
negating scores when switching between the AI and the opponent. The AI tries to
maximize its score while minimizing the opponent's score, but this is handled
by a single function using negative values.

core logic of Negamax with Alpha-Beta Pruning:

def negamax(is_ai, depth, alpha, beta):
    if game_win(list1) or game_win(list2) or depth == 0:
        return evaluation(is_ai)

    blank_list = list(set(list_all).difference(set(list3)))
    order(blank_list)  # Optimize the search order
    
    for next_step in blank_list:
        if not has_neightnor(next_step):
            continue

        if is_ai:
            list1.append(next_step)
        else:
            list2.append(next_step)
        list3.append(next_step)

        value = -negamax(not is_ai, depth - 1, -beta, -alpha)

        if value > alpha:
            alpha = value
            if depth == DEPTH:
                next_point[0] = next_step[0]
                next_point[1] = next_step[1]

            if alpha >= beta:
                return beta

    return alpha
This function alternates between maximizing the AI's score and minimizing the
opponent's score, updating the alpha and beta values to prune branches.

Key Code Concepts:
Evaluation Function:
The evaluation() function assigns a score to the current board state based
on predefined patterns (e.g., three in a row, four in a row), determining how
favorable the board is for the AI.


def evaluation(is_ai):
    # Calculate the score based on the current board state
Search Depth:
The depth of the search tree (DEPTH = 3 in this case) determines how far
ahead the AI looks. Deeper searches provide better moves but require more
computation.

Alpha-Beta Pruning Efficiency:
The order() function sorts the potential moves to improve the efficiency of
pruning. By exploring likely beneficial moves first, the algorithm can prune
unnecessary branches earlier.

def order(blank_list):
    last_pt = list3[-1]
    # Prioritize moves around the last point played
