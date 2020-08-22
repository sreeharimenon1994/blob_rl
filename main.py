from environment.base import Base
import json
from tqdm import tqdm
import math
import os

def main(cfg, model_path):
    """ main function to run everything
    """

    base = Base(epsilon=cfg['base']['epsilon'], eps_dec=cfg['base']['eps_dec'], padding=cfg['main']['padding'],\
                eps_min=cfg['base']['eps_min'], lr=cfg['base']['lr'], gamma=cfg['base']['gamma'],\
                w=cfg['main']['width'], h=cfg['main']['height'], batch_size=cfg['main']['batch_size'],\
                n_blobs=cfg['main']['n_blobs'], n_pheromones=cfg['base']['n_pheromones'], visualise=False,\
                n_steps=cfg["main"]["n_steps"], model_path=model_path, n_prev=cfg['base']['n_prev'])
    
    base.setup()

    for x in range(cfg["main"]["epochs"]):
        print('\nepoch: %d/%d' %(x, cfg["main"]["epochs"]))
        base.observation_aggregate()
        
        for y in tqdm(range(cfg["main"]["n_steps"])):
            base.step()
            base.learn()

        base.epsilon = max(0.01,  min(1.0, 1.0 - math.log10((x+1)/2)))
        base.reset()

    base.agent.save_model()


if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg = json.load(f)
        cfg['base']['epsilon'] += cfg['base']['n_prev'] * cfg['base']['eps_dec']
        f.close()

    # try:
    #     model_path = 'model/model.pt'
    #     model_path = os.path.dirname(os.path.abspath(model_path))
    #     print(model_path, 'model_path')
    # except:
    #     print('falied to load')
    model_path = ''
    main(cfg, model_path) 