def evaluate(players, agents, evaluators, eval_n_per_agent, game):
    print('evaluating...', end='')
    scores = []
    for e in evaluators:
        win = 0
        lose = 0
        for player, agent in zip(players, agents):
            agent.collecting = False
            for j in range(eval_n_per_agent):
                if j % 2 == 0:
                    r = game.run(e, player)
                else:
                    r = game.run(player, e)
                if r == e:
                    lose += 1
                elif r == player:
                    win += 1
                agent.finish_path()
            agent.collecting = True
        score = win/(win+lose)
        scores.append(score)
        print(score)
    return scores