from graphics import *
from PIL import Image as PILImage, ImageTk, ImageEnhance
import numpy as np

GRID_WIDTH = 40

COLUMN = 12
ROW = 12

list1 = []  # AI
list2 = []  # human
list3 = []  # all moves

list_all = []  # All points on the board
next_point = [0, 0]  # The position where AI should move next

ratio = 1  # Offensive coefficient, greater than 1 means offensive, less than 1 means defensive
DEPTH = 3  # Search depth, must be an odd number

shape_score = [(50, (0, 1, 1, 0, 0)),
               (50, (0, 0, 1, 1, 0)),
               (200, (1, 1, 0, 1, 0)),
               (500, (0, 0, 1, 1, 1)),
               (500, (1, 1, 1, 0, 0)),
               (5000, (0, 1, 1, 1, 0)),
               (5000, (0, 1, 0, 1, 1, 0)),
               (5000, (0, 1, 1, 0, 1, 0)),
               (5000, (1, 1, 1, 0, 1)),
               (5000, (1, 1, 0, 1, 1)),
               (5000, (1, 0, 1, 1, 1)),
               (5000, (1, 1, 1, 1, 0)),
               (5000, (0, 1, 1, 1, 1)),
               (50000, (0, 1, 1, 1, 1, 0)),
               (99999999, (1, 1, 1, 1, 1))]

def ai():
    global cut_count   # Record the number of alpha-beta pruning
    cut_count = 0
    global search_count   # Record the number of searches
    search_count = 0
    negamax(True, DEPTH, -99999999, 99999999)
    print("Number of prunings: " + str(cut_count))
    print("Number of searches: " + str(search_count))
    return next_point[0], next_point[1]

def negamax(is_ai, depth, alpha, beta):
    # Check if the game is over or if the search depth has reached the limit
    if game_win(list1) or game_win(list2) or depth == 0:
        return evaluation(is_ai)

    blank_list = list(set(list_all).difference(set(list3)))
    order(blank_list)   # Sort the search order to improve pruning efficiency
    # Traverse through each candidate move
    for next_step in blank_list:

        global search_count
        search_count += 1

        # If the evaluated position does not have adjacent moves, skip it to reduce computation
        if not has_neightnor(next_step):
            continue

        if is_ai:
            list1.append(next_step)
        else:
            list2.append(next_step)
        list3.append(next_step)

        value = -negamax(not is_ai, depth - 1, -beta, -alpha)
        if is_ai:
            list1.remove(next_step)
        else:
            list2.remove(next_step)
        list3.remove(next_step)

        if value > alpha:

            print(str(value) + "alpha:" + str(alpha) + "beta:" + str(beta))
            print(list3)
            if depth == DEPTH:
                next_point[0] = next_step[0]
                next_point[1] = next_step[1]
            # Alpha-beta pruning point
            if value >= beta:
                global cut_count
                cut_count += 1
                return beta
            alpha = value

    return alpha

# Neighboring positions to the last move are most likely to be optimal moves
def order(blank_list):
    last_pt = list3[-1]
    for item in blank_list:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if (last_pt[0] + i, last_pt[1] + j) in blank_list:
                    blank_list.remove((last_pt[0] + i, last_pt[1] + j))
                    blank_list.insert(0, (last_pt[0] + i, last_pt[1] + j))

def has_neightnor(pt):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if (pt[0] + i, pt[1]+j) in list3:
                return True
    return False

# Evaluation function
def evaluation(is_ai):
    total_score = 0

    if is_ai:
        my_list = list1
        enemy_list = list2
    else:
        my_list = list2
        enemy_list = list1

    # Calculate my score
    score_all_arr = []  # Positions of scoring shapes used to calculate overlapping score doubling
    my_score = 0
    for pt in my_list:
        m = pt[0]
        n = pt[1]
        my_score += cal_score(m, n, 0, 1, enemy_list, my_list, score_all_arr)
        my_score += cal_score(m, n, 1, 0, enemy_list, my_list, score_all_arr)
        my_score += cal_score(m, n, 1, 1, enemy_list, my_list, score_all_arr)
        my_score += cal_score(m, n, -1, 1, enemy_list, my_list, score_all_arr)

    # Calculate the enemy's score and subtract it
    score_all_arr_enemy = []
    enemy_score = 0
    for pt in enemy_list:
        m = pt[0]
        n = pt[1]
        enemy_score += cal_score(m, n, 0, 1, my_list, enemy_list, score_all_arr_enemy)
        enemy_score += cal_score(m, n, 1, 0, my_list, enemy_list, score_all_arr_enemy)
        enemy_score += cal_score(m, n, 1, 1, my_list, enemy_list, score_all_arr_enemy)
        enemy_score += cal_score(m, n, -1, 1, my_list, enemy_list, score_all_arr_enemy)

    total_score = my_score - enemy_score*ratio*0.1

    return total_score

# Score calculation in each direction
def cal_score(m, n, x_decrict, y_derice, enemy_list, my_list, score_all_arr):
    add_score = 0  # Additional score
    # Only take the maximum scoring shape in one direction
    max_score_shape = (0, None)

    # If there is already a scoring shape in this direction, do not calculate again
    for item in score_all_arr:
        for pt in item[1]:
            if m == pt[0] and n == pt[1] and x_decrict == item[2][0] and y_derice == item[2][1]:
                return 0

    # Check for scoring shapes in both directions from the current move
    for offset in range(-5, 1):
        pos = []
        for i in range(0, 6):
            if (m + (i + offset) * x_decrict, n + (i + offset) * y_derice) in enemy_list:
                pos.append(2)
            elif (m + (i + offset) * x_decrict, n + (i + offset) * y_derice) in my_list:
                pos.append(1)
            else:
                pos.append(0)
        tmp_shap5 = (pos[0], pos[1], pos[2], pos[3], pos[4])
        tmp_shap6 = (pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

        for (score, shape) in shape_score:
            if tmp_shap5 == shape or tmp_shap6 == shape:
                if tmp_shap5 == (1,1,1,1,1):
                    print('wwwwwwwwwwwwwwwwwwwwwwwwwww')
                if score > max_score_shape[0]:
                    max_score_shape = (score, ((m + (0+offset) * x_decrict, n + (0+offset) * y_derice),
                                               (m + (1+offset) * x_decrict, n + (1+offset) * y_derice),
                                               (m + (2+offset) * x_decrict, n + (2+offset) * y_derice),
                                               (m + (3+offset) * x_decrict, n + (3+offset) * y_derice),
                                               (m + (4+offset) * x_decrict, n + (4+offset) * y_derice)), (x_decrict, y_derice))

    # Calculate overlapping shapes. If two "three-lives" overlap, increase the score, except for a single stone
    if max_score_shape[1] is not None:
        for item in score_all_arr:
            for pt1 in item[1]:
                for pt2 in max_score_shape[1]:
                    if pt1 == pt2 and max_score_shape[0] > 10 and item[0] > 10:
                        add_score += item[0] + max_score_shape[0]

        score_all_arr.append(max_score_shape)

    return add_score + max_score_shape[0]

def game_win(list):
    for m in range(COLUMN):
        for n in range(ROW):

            if n < ROW - 4 and (m, n) in list and (m, n + 1) in list and (m, n + 2) in list and (
                    m, n + 3) in list and (m, n + 4) in list:
                return True
            elif m < ROW - 4 and (m, n) in list and (m + 1, n) in list and (m + 2, n) in list and (
                        m + 3, n) in list and (m + 4, n) in list:
                return True
            elif m < ROW - 4 and n < ROW - 4 and (m, n) in list and (m + 1, n + 1) in list and (
                        m + 2, n + 2) in list and (m + 3, n + 3) in list and (m + 4, n + 4) in list:
                return True
            elif m < ROW - 4 and n > 3 and (m, n) in list and (m + 1, n - 1) in list and (
                        m + 2, n - 2) in list and (m + 3, n - 3) in list and (m + 4, n - 4) in list:
                return True
    return False

def gobangwin():
    win = GraphWin("this is a gobang game", GRID_WIDTH * COLUMN, GRID_WIDTH * ROW)

    # Load background image and adjust transparency
    background_img = PILImage.open("image11.PNG").resize((GRID_WIDTH * COLUMN, GRID_WIDTH * ROW))
    enhancer = ImageEnhance.Brightness(background_img)
    background_img = enhancer.enhance(0.7)  # Set transparency to 0.7
    background_img.save("background_resized.png")  # Save adjusted image

    bg = Image(Point(GRID_WIDTH * COLUMN / 2, GRID_WIDTH * ROW / 2), "background_resized.png")
    bg.draw(win)

    i1 = 0
    while i1 <= GRID_WIDTH * COLUMN:
        l = Line(Point(i1, 0), Point(i1, GRID_WIDTH * COLUMN))
        l.setFill("black")
        l.setWidth(1)
        l.draw(win)
        i1 = i1 + GRID_WIDTH

    i2 = 0
    while i2 <= GRID_WIDTH * ROW:
        l = Line(Point(0, i2), Point(GRID_WIDTH * ROW, i2))
        l.setFill("black")
        l.setWidth(1)
        l.draw(win)
        i2 = i2 + GRID_WIDTH

    return win

def main():
    win = gobangwin()

    for i in range(COLUMN + 1):
        for j in range(ROW + 1):
            list_all.append((i, j))

    change = 0
    g = 0

    while g == 0:
        if change % 2 == 1:
            pos = ai()

            if pos in list3:
                message = Text(Point(200, 200), "Invalid position: " + str(pos[0]) + "," + str(pos[1]))
                message.draw(win)
                g = 1

            list1.append(pos)
            list3.append(pos)

            piece = Circle(Point(GRID_WIDTH * pos[0], GRID_WIDTH * pos[1]), 16)
            piece.setFill('white')
            piece.draw(win)

            if game_win(list1):
                message = Text(Point(100, 100), "White wins.")
                message.draw(win)
                g = 1
            change = change + 1

        else:
            p2 = win.getMouse()
            if not ((round((p2.getX()) / GRID_WIDTH), round((p2.getY()) / GRID_WIDTH)) in list3):

                a2 = round((p2.getX()) / GRID_WIDTH)
                b2 = round((p2.getY()) / GRID_WIDTH)
                list2.append((a2, b2))
                list3.append((a2, b2))

                piece = Circle(Point(GRID_WIDTH * a2, GRID_WIDTH * b2), 16)
                piece.setFill('black')
                piece.draw(win)

                if game_win(list2):
                    message = Text(Point(100, 100), "Black wins.")
                    message.draw(win)
                    g = 1

                change = change + 1

    message = Text(Point(100, 120), "Click anywhere to quit.")
    message.draw(win)
    win.getMouse()
    win.close()

main()
