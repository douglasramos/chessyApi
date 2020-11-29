import pandas as pd
import torch
import transformers
import time

import chess
import chess.pgn
import math

import transformers_model_utils
import transformers_model_execution

import model_input

from transformers import BertForSequenceClassification


device = torch.device("cpu")


def filter_testset(test):
    test = test[test['comment'].notnull()]
    return test


def fill_empty_comments_pred_label(df):
    segments_one = df[df['pred_label'] == 1]['segment_id'].unique().tolist()
    
    df.loc[df['pred_label'].isnull(), 'pred_label'] = df.loc[df['pred_label'].isnull()]\
                                .apply(lambda x: 1 if x['segment_id'] in segments_one else 0, axis=1)
    
    return df


def generate_input(pgn_file: str):
    game = chess.pgn.read_game(open(pgn_file))
    board = game.board()
    move_number = 1
    move_id = 0
    turn = 'W'

    id = []
    moves = []
    boards = []
    comments = []

    for node in game.mainline():
        # insere próximo movimento da partida no tabuleiro
        board.push(node.move)

        # tag para diferenciar jogada das brancas e das negras
        if move_number == math.floor(move_number):
            turn = 'W'
        else:
            turn = 'B'
        
        id.append(move_id)
        moves.append(f'{math.floor(move_number)}{turn}')
        boards.append(board.board_fen())
        comments.append(node.comment.replace("\n", " "))

        move_number += 0.5
        move_id += 1

    model_input = pd.DataFrame(columns=['id', 'move_number', 'fen', 'comment'])
    model_input['id'] = id
    model_input['move_number'] = moves
    model_input['fen'] = boards
    model_input['comment'] = comments

    return model_input


def generate_segment_ids(comments: list) -> list:
    ''' given a list of comments of each play on the chess game, it generates
    segment ids that groups certain plays on the game
    '''

    segment_ids = [0]
    current_id = 1 if comments[0] else 0

    for comment in comments[1:]:
        if(comment):
            segment_ids.append(current_id)
            current_id += 1

        else:
            segment_ids.append(current_id)

    return segment_ids


def predict(model_path, pgn_file, device):
    """chess_match: partida completa, mesmo com nan
    """
    start_time = time.time()

    chess_match = generate_input(pgn_file)
    chess_match['segment_id'] = generate_segment_ids(chess_match['comment'].tolist())
    chess_match['label'] = 0
    chess_match_filtered = chess_match[chess_match['comment'] != '']
    chess_match_filtered = chess_match_filtered[['id', 'comment', 'label']]

    model = BertForSequenceClassification.from_pretrained(model_path)
    prediction_dataloader = transformers_model_utils.load_test_set(chess_match_filtered)

    flat_predictions, _ = transformers_model_execution.run_model(model,
                                                                 prediction_dataloader,
                                                                 device)
    chess_match_filtered['pred_label'] = flat_predictions

    full_match = chess_match.merge(chess_match_filtered, how='left', on=['id', 'comment', 'label'])
    full_match = fill_empty_comments_pred_label(full_match)

    end_time = time.time()
    print(end_time - start_time)

    full_match = full_match.drop(columns='label')
    full_match.to_csv('pgn_prediction.csv', index=False)

    return full_match


if __name__ == '__main__':
    predict('transformers_model/', 'pgn_256.txt', device)
