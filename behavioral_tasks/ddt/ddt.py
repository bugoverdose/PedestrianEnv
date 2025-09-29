import os
import sys

from pathlib import Path
import numpy as np
import pandas as pd

from datetime import datetime
from psychopy import core, visual, event
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from adopy.tasks.dd import TaskDD, ModelHyp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine import Engine

import argparse

from scipy.stats import bernoulli

from adopy import Task, Model

PATH_ROOT = Path(__file__).absolute().parent
PATH_IMAGE = PATH_ROOT / 'images'

class DdtDrawer:
    """PsychoPy drawer for the choice under risk and ambiguity task."""

    def __init__(self,
                 window: visual.Window,
                 box_w: float = 6,
                 box_h: float = 6,
                 dist_btwn: float = 8,
                 text_font: str = 'Nanum Gothic',
                 text_size: int = 1,
                 text_margin: float = 1,
                 fixation_size: float = 1,
                 pos_x: float = 0,
                 pos_y: float = 0,
                 ):
        self.window = window
        self.box_w = box_w
        self.box_h = box_h
        self.dist_btwn = dist_btwn
        self.text_font = text_font
        self.text_size = text_size
        self.text_margin = text_margin
        self.fixation_size = fixation_size
        self.pos_x = pos_x
        self.pos_y = pos_y

        with open(PATH_ROOT / 'instructions_ddt.yml', 'r', encoding='utf-8') as f:
            self.instructions = yaml.load(f, Loader=Loader)

    @staticmethod
    def convert_delay_to_str(delay):
        tbl_conv = {
            0: '바로 받기',
            0.43: '3일 후에 받기',
            0.714: '5일 후에 받기',
            1: '1주 후에 받기',
            2: '2주 후에 받기',
            3: '3주 후에 받기',
            4.3: '1달 후에 받기',
            6.44: '6주 후에 받기',
            8.6: '2달 후에 받기',
            10.8: '10주 후에 받기',
            12.9: '3달 후에 받기',
            17.2: '4달 후에 받기',
            21.5: '5달 후에 받기',
            26: '6달 후에 받기',
            52: '1년 후에 받기',
            104: '2년 후에 받기',
            156: '3년 후에 받기',
            260: '5년 후에 받기',
            520: '10년 후에 받기'
        }
        mv, ms = None, None
        for (v, s) in tbl_conv.items():
            if mv is None or np.square(delay - mv) > np.square(delay - v):
                mv, ms = v, s
        return ms

    def draw_option(self, delay, amount, direction=1, chosen=False):
        pos_x_center = self.pos_x + direction * self.dist_btwn
        pos_x_left = pos_x_center - self.box_w
        pos_x_right = pos_x_center + self.box_w
        pos_y_top = self.pos_y + self.box_h / 2
        pos_y_bottom = self.pos_y - self.box_h / 2

        vertices = (
            (pos_x_left, pos_y_top),
            (pos_x_right, pos_y_top),
            (pos_x_right, pos_y_bottom),
            (pos_x_left, pos_y_bottom)
        )

        box = visual.ShapeStim(
            self.window,
            lineWidth=12 if chosen else 8,
            lineColor='white',
            fillColor='darkgreen' if chosen else 'black',
            vertices=vertices,
        )
        box.draw()

        text_a = visual.TextStim(
            self.window,
            '₩ {:,.0f}'.format(amount * 1000),
            font=self.text_font,
            pos=(pos_x_center, self.pos_y + self.text_margin),
        )
        text_a.size = self.text_size
        text_a.draw()

        text_d = visual.TextStim(
            self.window,
            self.convert_delay_to_str(delay),
            font=self.text_font,
            pos=(pos_x_center, self.pos_y - self.text_margin),
        )
        text_d.size = self.text_size
        text_d.draw()

    def draw(self, t_ss, t_ll, r_ss, r_ll, direction, chosen=None):
        if direction == 1:
            d_ss, d_ll = -1, 1
        else:
            d_ss, d_ll = 1, -1

        self.draw_option(t_ss, r_ss, d_ss, chosen == 1)  # SS
        self.draw_option(t_ll, r_ll, d_ll, chosen == 0)  # LL

    def draw_fixation(self):
        fixation = visual.GratingStim(
            self.window,
            color='white',
            tex=None,
            mask='cross',
            size=self.fixation_size,
        )
        fixation.draw()

    def draw_button_to_proceed(self):
        btn = visual.ImageStim(
            self.window,
            str((PATH_IMAGE / 'button-to-proceed.png').absolute()),
            pos=(0, -self.box_h / 2 - 4 * self.text_margin),
            size=(9.43, 2))
        btn.draw()

    def draw_intro(self):
        text = visual.TextStim(
            self.window,
            self.instructions['intro'],
            font=self.text_font,
            pos=(0, 0),
            wrapWidth=30,
            anchorVert='center')
        text.draw()
        self.draw_button_to_proceed()

    def draw_train_before(self, page, n_trial):
        if page == 0:
            tmppos = (self.pos_x, self.pos_y)
            self.pos_x, self.pos_y = 0, -2 * self.text_margin
            self.draw(0, 52, 100, 800, 1)
            self.pos_x, self.pos_y = tmppos

            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][0].format(n_trial),
                font=self.text_font,
                pos=(0, self.box_h / 2 + 1 * self.text_margin),
                wrapWidth=30,
            )
            text.size = self.text_size
            text.draw()

        elif page == 1:
            tmppos = (self.pos_x, self.pos_y)
            self.pos_x, self.pos_y = 0, -2 * self.text_margin
            self.draw(0, 52, 100, 800, 1)
            self.pos_x, self.pos_y = tmppos

            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][1],
                font=self.text_font,
                pos=(0, self.box_h / 2 + 1 * self.text_margin),
                wrapWidth=30,
            )
            text.size = self.text_size
            text.draw()

        elif page == 2:
            tmppos = (self.pos_x, self.pos_y)
            self.pos_x, self.pos_y = 0, -2 * self.text_margin
            self.draw(0, 52, 100, 800, 1)
            self.pos_x, self.pos_y = tmppos

            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][2],
                font=self.text_font,
                pos=(0, self.box_h / 2 + 1 * self.text_margin),
                wrapWidth=30,
            )
            text.size = self.text_size
            text.draw()

        elif page == 3:
            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][3],
                font=self.text_font,
                pos=(0, 0),
                wrapWidth=30,
            )
            text.size = self.text_size
            text.draw()
        
        else:
            raise ValueError('Invalid page number for instructions.')
        self.draw_button_to_proceed()

    def draw_train_after(self):
        text = visual.TextStim(
            self.window,
            self.instructions['train_after'],
            font=self.text_font,
            pos=(0, 0),
            wrapWidth=30,
            anchorVert='center')
        text.draw()
        self.draw_button_to_proceed()

    def draw_main_before(self, n_trial):
        text = visual.TextStim(
            self.window,
            self.instructions['main_before'].format(n_trial),
            font=self.text_font,
            pos=(0, 0),
            wrapWidth=30,
            anchorVert='center')
        text.draw()
        self.draw_button_to_proceed()

    def draw_main_after(self):
        text = visual.TextStim(
            self.window,
            self.instructions['main_after'],
            font=self.text_font,
            pos=(0, 0),
            wrapWidth=30,
            anchorVert='center')
        text.draw()
        self.draw_button_to_proceed()

    def draw_outro(self):
        text = visual.TextStim(
            self.window,
            self.instructions['outro'],
            font=self.text_font,
            pos=(0, 0),
            wrapWidth=30,
            anchorVert='center')
        text.draw()

def make_grid(start, end, n):
    return np.linspace(start, end, n + 1, endpoint=False)[1:]

def make_grid_log(a, b, n):
    return np.logspace(np.log10(a), np.log10(b), n + 1, endpoint=False)[1:]

class DdtRunner:
    def __init__(self, window, drawer, subj, path_output):
        self.window = window
        self.drawer = drawer
        self.subj = subj
        self.path_output = path_output
        if os.path.exists(path_output):
            self.df = pd.read_table(path_output, sep='\t')
        else:
            self.df = pd.DataFrame(None)

        self.task = TaskDD()
        self.model = ModelHyp()
        self.grid_response = {'choice': [0, 1]}
        self.engine = Engine(
            task=self.task,
            model=self.model,
            grid_design=self.generate_grid_designs(),
            grid_param=self.generate_grid_params(),
            grid_response=self.grid_response)

        self.task_log = Task(
        name='Delay Discounting Task',
        designs=['t_ss', 't_ll', 'r_ss', 'r_ll'],
        responses=['choice'])

        self.model_log = Model(
        name='Hyperbolic',
        task=self.task_log,  
        params=['log_k', 'tau_2'],
        func=self.compute_prob_log)

    def _initialize_engine(self):
        engine = Engine(
        task=self.task,
        model=self.model,
        grid_design=self.generate_grid_designs(),
        grid_param=self.generate_grid_params(),
        grid_response=self.grid_response)
        return engine
    
    def _initialize_engine_log(self):
        engine_log = Engine(
        task=self.task_log,
        model=self.model_log,
        grid_design=self.generate_grid_designs(),
        grid_param=self.generate_grid_params_log(),
        grid_response=self.grid_response)
        return engine_log
    
    def save_record(self):
        columns=[
            'subject', 'block', 'block_type', 'trial',
            *(self.task.designs),'resp_ss', 'rt', 'lr',
            *['mean_' + p for p in self.model.params],
            *['sd_' + p for p in self.model.params],
            *['mean_' + p for p in self.model_log.params],
            *['sd_' + p for p in self.model_log.params],
        ]
        self.df[columns].to_csv(self.path_output, sep='\t', index=False)

    @staticmethod
    def generate_grid_designs():
        # Amounts of rewards
        am_soon = np.arange(10, 800, 10)
        am_late = [800]

        amounts = np.vstack([(ams, aml) for ams in am_soon for aml in am_late if ams < aml])

        # Delays
        t_ss = [0]
        t_ll = [
            0.43, 0.714, 1, 2, 3,
            4.3, 6.44, 8.6, 10.8, 12.9,
            17.2, 21.5, 26, 52, 104,
            156, 260, 520
        ]

        delays = np.vstack([
            (ds, dl) for ds in t_ss for dl in t_ll if ds < dl
        ])

        designs = {('t_ss', 't_ll'): delays, ('r_ss', 'r_ll'): amounts}
        return designs

    @staticmethod
    def generate_grid_params():
        # k: discounting rate parameter
        k = np.logspace(0,5,101)/100_000 # range: 0.00001 ~ 1.00000
        # tau: inverse temperature parameter
        tau = np.linspace(0,1,101)[1:] # range: 0.01 ~ 1.00
        params = {'k': k, 'tau': tau}
        return params
    
    ### For estimating log k
    @staticmethod
    def compute_prob_log(choice, t_ss, t_ll, r_ss, r_ll, log_k, tau_2):
        SV_ss = r_ss * (1 / (1 + np.exp(log_k) * t_ss))
        SV_ll = r_ll * (1 / (1 + np.exp(log_k) * t_ll))
            
        #print(SV_ss, SV_ll, tau)
        prob_ll = 1 / (1 + np.exp(-tau_2 * (SV_ll - SV_ss)))
    
        return bernoulli.logpmf(choice, prob_ll) # return log of the probability mass function
    
    @staticmethod
    def generate_grid_params_log():
        # k: discounting rate parameter
        log_k = np.linspace(-11.51292546, 0, 101) # natural log # np.linspace(-5, 0, 101) 
        # tau: inverse temperature parameter
        tau_2 = np.linspace(0,1,101) #make_grid(0, 10, 30)
        params = {'log_k': log_k, 'tau_2': tau_2}
        return params

    def show_countdown(self):
        text1 = visual.TextStim(self.window, text="1",
                                pos=(0.0, 0.0), height=2)
        text2 = visual.TextStim(self.window, text="2",
                                pos=(0.0, 0.0), height=2)
        text3 = visual.TextStim(self.window, text="3",
                                pos=(0.0, 0.0), height=2)

        text3.draw()
        self.window.flip()
        core.wait(1)

        text2.draw()
        self.window.flip()
        core.wait(1)

        text1.draw()
        self.window.flip()
        core.wait(1)

    def show_intro(self):
        self.drawer.draw_intro()
        self.window.flip()
        _ = event.waitKeys(keyList=['space', 'return'])

    def show_outro(self):
        self.drawer.draw_outro()
        self.window.flip()
        _ = event.waitKeys(keyList=['space', 'return'])

    def show_block_start(self, n_trial):
        self.drawer.draw_main_before(n_trial)
        self.window.flip()
        _ = event.waitKeys(keyList=['space', 'return'])

    def show_block_end(self, block):
        self.drawer.draw_main_after()
        self.window.flip()
        _ = event.waitKeys(keyList=['space', 'return'])

    def fixed_design(self, n_trial):
        #direction: balanced random
        half_trial = n_trial//2
        lr_design = [-1]*half_trial+[1]*(n_trial-half_trial)
        np.random.shuffle(lr_design)

        return lr_design

    def run_trial(self, design, lr):
        self.drawer.draw_fixation()
        self.window.flip()
        core.wait(1)

        self.drawer.draw(design['t_ss'], design['t_ll'],
                         design['r_ss'], design['r_ll'],
                         lr)
        self.window.flip()
        timer = core.Clock()
        keys = event.waitKeys(keyList=['s', 'l', 'escape'])
        rt = timer.getTime()

        if keys[0] == 'escape':
            core.quit()

        resp_left = int(keys[0] == 's')
        # 1 if chosen left, 0 if chosen right option
        resp_ss = int((1 - resp_left) ^ (lr == 1))
        # 1 if chosen SS option, 0 if chosen LL option
        self.drawer.draw(design['t_ss'], design['t_ll'],
                         design['r_ss'], design['r_ll'],
                         lr, resp_ss)
        self.window.flip()
        core.wait(1)

        return resp_ss, rt

    def run_train_block(self, n_train_trial=4, n_trial=20):
        """Run a block for training purpose."""
        for i in range(4):
            self.drawer.draw_train_before(i,  n_trial)
            self.window.flip()
            _ = event.waitKeys(keyList=['space', 'return'])
        lr_design = self.fixed_design(n_trial)
        self.show_countdown()
        for trial in range(n_train_trial):
            design = self.engine.get_design('random')
            # print(trial, design)
            _ = self.run_trial(design, lr_design[trial])

        self.drawer.draw_train_after()
        self.window.flip()
        _ = event.waitKeys(keyList=['space', 'return'])

    def run_block(self, block, block_type, n_trial=20):
        """Run a block with optimal designs chosen by ADO."""
        self.engine = self._initialize_engine()
        self.engine_log = self._initialize_engine_log()
        self.engine.reset()
        self.engine_log.reset()

        lr_design = self.fixed_design(n_trial)

        self.show_countdown()
        for trial in range(n_trial):
            if block_type == 'ado':
                design = self.engine.get_design('optimal')
            else:
                raise Exception("block_type 'ado' only allowed")
            lr = lr_design[trial]
            resp_ss, rt = self.run_trial(design, lr)
            resp_ll = 1 - resp_ss
            self.engine.update(design, resp_ll)
            self.engine_log.update(design, resp_ll)
            
            dict_mean = {
                'mean_' + p: m
                for p, m in zip(self.model.params, self.engine.post_mean)
            }
            dict_sd = {
                'sd_' + p: m
                for p, m in zip(self.model.params, self.engine.post_sd)
            }

            dict_mean_log = {
                'mean_' + p: m
                for p, m in zip(self.model_log.params, self.engine_log.post_mean)
            }

            dict_sd_log = {
                'sd_' + p: m
                for p, m in zip(self.model_log.params, self.engine_log.post_sd)
            }

            self.df = pd.concat([self.df, pd.DataFrame([{
                'subject': self.subj,
                'block': block,
                'block_type': block_type,
                'trial': trial + 1,
                **design,
                'resp_ss': resp_ss,
                'rt': rt,
                'lr': lr,
                **dict_mean,
                **dict_sd,
                **dict_mean_log,
                **dict_sd_log
            }])], ignore_index=True)

            self.save_record()

def run_ddt_ado(subj, window):
    n_block = 1
    n_trial = 20
    n_train_trial = 4
    has_tutorial = True

    time_now = datetime.now()
    time_now_iso = time_now.isoformat().replace(':', '-').replace('T', '-')[:-7]
    
    # Save Data
    path_data = PATH_ROOT.parent.parent / 'data' / f"{subj}"
    fn_data = f"DDT_{time_now_iso}.csv"
    path_data.mkdir(exist_ok=True)
    path_output = str(path_data / fn_data)

    block_types = ['ado'] * n_block
    print('Block types:', block_types)

    # Initialize a drawer and a runner
    drawer = DdtDrawer(window)
    runner = DdtRunner(window, drawer, subj, path_output)

    # Run blocks
    if has_tutorial:
        runner.show_intro()
        runner.run_train_block(n_train_trial, n_trial)

    for block, block_type in enumerate(block_types):
        runner.show_block_start(n_trial)
        runner.run_block(block + 1, block_type, n_trial)

        if block+1 != n_block:
            runner.show_block_end(block + 1)
        else: 
            runner.show_outro()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Run for experiment")
    arg_parser.add_argument('--subjId', type=int, default=1, help='subject ID')
    args = arg_parser.parse_args()

    # Open a window
    window = visual.Window(size=[1512, 982],
                           units='deg',
                           monitor='testMonitor',
                           color="black",
                           screen=0,
                           allowGUI=False,
                           fullscr=True) 
    event.globalKeys.clear()
    event.globalKeys.add(key='escape', func=core.quit, name='shutdown')

    run_ddt_ado(args.subjId, window)
