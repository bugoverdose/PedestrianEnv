import os
import sys

from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd

from datetime import datetime
from psychopy import core, visual, event
import yaml

from adopy.tasks.cra import TaskCRA, ModelLinear
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine import Engine

import argparse

PATH_ROOT = Path(__file__).absolute().parent

PATH_IMAGE = PATH_ROOT / 'images'

class State(Enum):
    """States for a option in the CRA task.

    There are three states:
    1. **Inactive**: an option is shown but the response is not accepted.
    2. **Active**
    3. **Hidden**
    4. **Chosen**
    """
    inactive = 0
    active = 1
    hidden = 2
    chosen = 3

class Color(Enum):
    """Colors for the choice under risk and ambiguity task."""
    high_inactive = '#9b001f'
    high_active = '#ff0033'
    low_inactive = '#00098f'
    low_active = '#0010ff'
    ambig_inactive = '#585858'
    ambig_active = '#939393'
    border_chosen = '#ffffff'
    border = 'black'
    background = 'black' #'#333333'

class Direction(Enum):
    """Directions for options of the choice under risk and ambiguity task."""
    left = -1
    right = 1

    @staticmethod
    def reverse(x):
        if x is Direction.left:
            return Direction.right
        elif x is Direction.right:
            return Direction.left
        raise TypeError('Direction object is needed.')

    @staticmethod
    def random(lr):
        if lr == -1: #random.random() > 0.5:
            return Direction.left
        else:
            return Direction.right

class CraDrawer:
    """PsychoPy drawer for the choice under risk and ambiguity task."""

    def __init__(self,
                 window: visual.Window,
                 box_w: float = 2, #originally 3
                 box_h: float = 8, #originally 12
                 dist_btwn: float = 16,
                 text_font: str = 'Nanum Gothic',
                 text_size: int = 1, #originally 13
                 text_margin: float = 0.5,
                 fixation_size: float = 0.8, #originally 1
                 ):
        self.window = window
        self.box_w = box_w
        self.box_h = box_h
        self.dist_btwn = dist_btwn
        self.text_font = text_font
        self.text_size = text_size
        self.text_margin = text_margin
        self.fixation_size = fixation_size

        with open(PATH_ROOT / 'instructions_cra.yml', 'r', encoding='utf-8') as f:
            self.instructions = yaml.load(f, Loader=yaml.FullLoader)

    def draw_box_high(self, state: State, lr: Direction, prob: float = 0.5):
        """Draw a box for the high reward."""
        assert lr in Direction
        assert state in State
        assert state is not State.hidden
        assert 0 <= prob <= 1

        # Configs for the box
        linewidth = {
            State.inactive: 6,
            State.active: 6,
            State.chosen: 8,
        }
        linecolor = {
            State.inactive: Color.border.value,
            State.active: Color.border.value,
            State.chosen: Color.border_chosen.value,
        }
        fillcolor = {
            State.inactive: Color.high_inactive.value,
            State.active: Color.high_active.value,
            State.chosen: Color.high_active.value,
        }

        box_x_center = lr.value * self.dist_btwn / 2
        box_x_left = box_x_center - self.box_w
        box_x_right = box_x_center + self.box_w
        box_y_top = self.box_h / 2
        box_y_bottom = box_y_top - self.box_h * prob

        vertices = ((box_x_left, box_y_top),
                    (box_x_right, box_y_top), 
                    (box_x_right, box_y_bottom),
                    (box_x_left, box_y_bottom))

        # Draw the box
        box = visual.ShapeStim(
            self.window,
            lineWidth=linewidth[state],
            lineColor=linecolor[state],
            fillColor=fillcolor[state],
            vertices=vertices,
        )
        box.draw()

    def draw_box_low(self, state: State, lr: Direction, prob: float = 0.5):
        assert lr in Direction
        assert state in State
        assert state is not State.hidden
        assert 0 <= prob <= 1

        # Configs for the box
        linewidth = {
            State.inactive: 6,
            State.active: 6,
            State.chosen: 8,
        }
        linecolor = {
            State.inactive: Color.border.value,
            State.active: Color.border.value,
            State.chosen: Color.border_chosen.value,
        }
        fillcolor = {
            State.inactive: Color.low_inactive.value,
            State.active: Color.low_active.value,
            State.chosen: Color.low_active.value,
        }

        box_x_center = lr.value * self.dist_btwn / 2
        box_x_left = box_x_center - self.box_w
        box_x_right = box_x_center + self.box_w
        box_y_bottom = -self.box_h / 2
        box_y_top = box_y_bottom + self.box_h * (1 - prob)

        vertices = ((box_x_left, box_y_top),
                    (box_x_right, box_y_top),
                    (box_x_right, box_y_bottom),
                    (box_x_left, box_y_bottom))

        # Draw the box
        box = visual.ShapeStim(
            self.window,
            lineWidth=linewidth[state],
            lineColor=linecolor[state],
            fillColor=fillcolor[state],
            vertices=vertices,
        )
        box.draw()

    def draw_box_ambig(self, state: State, lr: Direction, ambig: float):
        assert lr in Direction
        assert state in State
        assert state is not State.hidden
        assert 0 <= ambig <= 1

        if ambig == 0:
            return

        # Configs for the box
        linewidth = {
            State.inactive: 0,
            State.active: 0,
            State.chosen: 8,
        }
        linecolor = {
            State.inactive: Color.border.value,
            State.active: Color.border.value,
            State.chosen: Color.border_chosen.value,
        }
        fillcolor = {
            State.inactive: Color.ambig_inactive.value,
            State.active: Color.ambig_active.value,
            State.chosen: Color.ambig_active.value,
        }

        box_x_center = lr.value * self.dist_btwn / 2
        box_x_left = box_x_center - self.box_w * 1.1
        box_x_right = box_x_center + self.box_w * 1.1
        box_y_top = ambig * self.box_h / 2
        box_y_bottom = -box_y_top

        vertices = ((box_x_left, box_y_top),
                    (box_x_right, box_y_top),
                    (box_x_right, box_y_bottom),
                    (box_x_left, box_y_bottom))

        # Draw the box
        box = visual.ShapeStim(
            self.window,
            lineWidth=linewidth[state],
            lineColor=linecolor[state],
            fillColor=fillcolor[state],
            vertices=vertices,
        )
        box.draw()

    def draw_box_fixed(self, state: State, lr: Direction):
        self.draw_box_high(state, lr, 0.5)
        self.draw_box_low(state, lr, 0.5)
    #fixed option: fixed to 50:50 chance

    def draw_box_variable(self, state: State, lr: Direction,
                          prob: float, ambig: float):
        self.draw_box_high(state, lr, prob)
        self.draw_box_low(state, lr, prob)
        self.draw_box_ambig(state, lr, ambig)
    #risky op

    def draw_text_high(self, lr: Direction, reward: float = 0):
        pos = (lr.value * self.dist_btwn / 2,
               self.box_h / 2 + self.text_margin)
        text = visual.TextStim(self.window,
                               text='₩ {:,.0f}'.format(reward * 1000),
                               pos=pos)
        text.size = self.text_size
        text.draw()

    def draw_text_low(self, lr: Direction, reward: float = 0):
        pos = (lr.value * self.dist_btwn / 2,
               -(self.box_h / 2 + self.text_margin))
        text = visual.TextStim(self.window,
                               text='₩ {:,.0f}'.format(reward * 1000),
                               pos=pos)
        text.size = self.text_size
        text.draw()

    def draw_fixed(self, state: State, lr: Direction, reward: float, color):
        if color == 0:
            reward_low = reward
            reward_high = 0
        else:
            reward_low = 0
            reward_high = reward
        self.draw_box_fixed(state, lr)
        self.draw_text_high(lr, reward_high)
        self.draw_text_low(lr, reward_low)

    def draw_variable(self, state: State, lr: Direction, prob: float,
                      ambig: float, reward: float, color):
        #p_var = 1-p_var if blue
        if color == 0:
            prob = 1-prob
            reward_low = reward
            reward_high = 0
        else:
            reward_low = 0
            reward_high = reward

        self.draw_box_variable(state, lr, prob, ambig)
        self.draw_text_high(lr, reward_high)
        self.draw_text_low(lr, reward_low)

    def draw(self,
             state: State,
             lr: Direction,
             p_var: float,
             a_var: float,
             r_var: float,
             r_fix: float,
             color):
        """Draw two options based on given design values.

        Parameters
        ----------
        state : State
            Current state for two options
        lr : Direction
            Direction to show a fixed option
        p_var : float
            Probability to win for a variable option
        a_var : float
            Level of ambiguity for a variable option
        r_var : float
            Probabilistic reward to win for a variable option
        r_fix : float
            Probabilistic reward to win for a variable option
        """
        self.draw_fixed(state, lr, r_fix, color)
        lr_rev = Direction.reverse(lr)
        self.draw_variable(state, lr_rev, p_var, a_var, r_var, color)

    def draw_one(self,
                 state: State,
                 lr: Direction,
                 lr_chosen: Direction,
                 p_var: float,
                 a_var: float,
                 r_var: float,
                 r_fix: float,
                 color):
        """Draw only one option based on the choice a subject makes.

        Parameters
        ----------
        state : State
            Current state for two options
        lr : Direction
            Direction to show a fixed option
        lr_chosen : Direction
            Direction a subject chooses
        p_var : float
            Probability to win for a variable option
        a_var : float
            Level of ambiguity for a variable option
        r_var : float
            Probabilistic reward to win for a variable option
        r_fix : float
            Probabilistic reward to win for a variable option
        """
        if lr is lr_chosen:
            self.draw_fixed(state, lr, r_fix, color)
        else:
            lr_rev = Direction.reverse(lr)
            self.draw_variable(state, lr_rev, p_var, a_var, r_var, color)

    def draw_fixation(self):
        fixation = visual.GratingStim(self.window,
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

    def draw_train_before(self, page=6):
        if page == 0:
            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][0],
                font=self.text_font,
                pos=(0, self.box_h / 2 + 3 * self.text_margin),
                wrapWidth=30,
                anchorVert='bottom')
            text.draw()
            self.draw(State.active, Direction.left, 0.3, 0, 34, 5, 1)

        elif page == 1:
            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][1],
                font=self.text_font,
                pos=(0, self.box_h / 2 + 3 * self.text_margin),
                wrapWidth=30,
                anchorVert='bottom')
            text.draw()

            self.draw(State.active, Direction.left, 0.3, 0, 34, 5, 1)

        elif page == 2:
            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][2],
                font=self.text_font,
                pos=(self.dist_btwn / 2, 0),
                wrapWidth=20,
                anchorVert='center')
            text.draw()

            self.draw_fixed(State.active, Direction.left, 5, 0)

        elif page == 3:
            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][3],
                font=self.text_font,
                pos=(-self.dist_btwn / 2, 0),
                wrapWidth=20,
                anchorVert='center')
            text.draw()

            self.draw_variable(State.active, Direction.right, 0.5, 0.5, 34, 0)

        elif page == 4:
            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][4],
                font=self.text_font,
                pos=(0, self.box_h / 2),
                wrapWidth=30,
                anchorVert='bottom')
            text.draw()

            img = visual.ImageStim(
                self.window,
                str((PATH_IMAGE / 'choice.png').absolute()),
                pos=(0, 0),
                size=(30, 10.43))
            img.draw()

        elif page == 5:
            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][5],
                font=self.text_font,
                pos=(0, self.box_h / 2),
                wrapWidth=30,
                anchorVert='bottom')
            text.draw()

            img = visual.ImageStim(
                self.window,
                str((PATH_IMAGE / 'choice.png').absolute()),
                pos=(0, 0),
                size=(30, 10.43))
            img.draw()

        elif page == 6:
            text = visual.TextStim(
                self.window,
                self.instructions['train_before'][6],
                font=self.text_font,
                pos=(0, 0),
                wrapWidth=30,
                anchorVert='center')
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

    def draw_main_before(self, block, n_trial):
        text = visual.TextStim(
            self.window,
            self.instructions['main_before'].format(n_trial),
            font=self.text_font,
            pos=(0, 0),
            wrapWidth=30,
            anchorVert='center')
        text.draw()
        self.draw_button_to_proceed()

    def draw_main_after(self, block):
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

class CraRunner:
    def __init__(self, window, drawer, subj, path_output):
        self.window = window
        self.drawer = drawer
        self.subj = subj
        self.path_output = path_output
        if os.path.exists(path_output):
            self.df = pd.read_table(path_output, sep='\t')
        else:
            self.df = pd.DataFrame(None)

        self.task = TaskCRA()
        self.model = ModelLinear()
        self.grid_response = {'choice': [0, 1]}
        self.engine = Engine(
            task=self.task,
            model=self.model,
            grid_design=self.generate_grid_designs(),
            grid_param=self.generate_grid_params(),
            grid_response=self.grid_response)

    def save_record(self):
        columns=[
            'subject', 'block', 'block_type', 'trial',
            *(self.task.designs), 'resp_var', 'rt', 'color', 'lr',
            *['mean_' + p for p in self.model.params],
            *['sd_' + p for p in self.model.params],
        ]
        self.df[columns].to_csv(self.path_output, sep='\t', index=False)

    @staticmethod
    def generate_grid_designs():
        pval = [0.13, 0.25, 0.38]
        aval = [0.25, 0.50, 0.75]

        pa_risky = [[p, 0] for p in pval if 0 < p < 0.5]
        pa_ambig = [[0.5, a] for a in aval if 0 < a]

        pr_am = np.array(pa_risky + pa_ambig)

        rval = [5, 6, 7, 8, 9.5,
                11, 13, 15, 18, 21,
                25, 29, 34, 40, 47,
                55, 65]  # 17 points
        rs = np.vstack([(rv, rf) for rv in rval for rf in rval if rv > rf]+ [(5, 5)])
        # rs = np.vstack([(rv, rf) for rv in rval for rf in rval if rv > rf])
        designs = {('p_var', 'a_var'): pr_am, ('r_var', 'r_fix'): rs}
        return designs
        # p_var: proportion of risky box; a_var: proportion of ambiguous box
        # r_var: reward of variable option (either risky or ambiguous)
        # r_fix: reward of fixed option (fixed to 5:5)
        

    @staticmethod
    def generate_grid_params():
        alp = np.linspace(0, 2, 41)[1:]
        bet = np.linspace(-3, 3, 41)
        gam = np.linspace(0, 5, 21) #np.linspace(0, 5, 21)[1:]

        # alpha: risk attitude parameter
        # beta: ambiguity attitude parameter
        # gamma: inverse temperature

        params = {'alpha': alp, 'beta': bet, 'gamma': gam}
        return params

    @staticmethod
    def generate_fixed_designs():
        """Return design pairs used by Levy et al. (2009)"""
        # For risky conditions
        pr_risky = np.array([.13, .25, .38])
        am_risky = np.array([.0])

        # For ambiguous conditions
        pr_ambig = np.array([.5])
        am_ambig = np.array([.25, .50, .75])

        # Make cartesian products for each condition
        pr_am_risky = np.squeeze(np.stack(np.meshgrid(pr_risky, am_risky), -1))
        pr_am_ambig = np.squeeze(np.stack(np.meshgrid(pr_ambig, am_ambig), -1))

        # Merge two grids into one object
        pr_am = np.vstack([pr_am_risky, pr_am_ambig])

        rv = np.array([5, 9.5, 18, 34, 65]) # rewards for variable option
        rf = np.array([5])                  # rewards for option fixed to 50% chance

        rws = np.vstack([(v, f) for v in rv for f in rf])

        designs = np.array([
            np.concatenate([pr_am[i], rws[j], [k]])
            for i in range(len(pr_am))
            for j in range(len(rws))
            for k in range(2)
        ])
        np.random.shuffle(designs)

        return pd.DataFrame(designs, columns=[
            'p_var', 'a_var', 'r_var', 'r_fix', 'color'])

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
        _ = event.waitKeys(keyList=['space', 'escape'])

    def show_block_start(self, block, n_trial):
        self.drawer.draw_main_before(block, n_trial)
        self.window.flip()
        _ = event.waitKeys(keyList=['space'])

    def show_block_end(self, block):
        self.drawer.draw_main_after(block)
        self.window.flip()
        _ = event.waitKeys(keyList=['space'])

    def run_trial(self, design, color, lr_fixed):
        self.drawer.draw_fixation()
        self.window.flip()
        core.wait(1)

        lr = Direction.random(lr_fixed)
        self.drawer.draw(State.inactive, lr,
                         design['p_var'], design['a_var'],
                         design['r_var'], design['r_fix'],color)
        self.window.flip()
        core.wait(1)

        self.drawer.draw(State.active, lr,
                         design['p_var'], design['a_var'],
                         design['r_var'], design['r_fix'],color)
        self.window.flip()
        timer = core.Clock()
        keys = event.waitKeys(keyList=['s', 'l', 'escape'])
        rt = timer.getTime()
        
        if keys[0] == 'escape':
            core.quit()

        lr_chosen = Direction.left if keys[0] == 's' else Direction.right
        self.drawer.draw_one(State.chosen, lr, lr_chosen,
                             design['p_var'], design['a_var'],
                             design['r_var'], design['r_fix'],color)
        self.window.flip()
        core.wait(1)

        resp = int(lr is not lr_chosen)
        # lr = fixed option direction
        # 1 if chosen variable option, 0 if chosen fixed option

        return resp, rt

    def run_train_block(self, n_trial=4):
        """Run a block for training purpose."""
        for i in range(7):
            self.drawer.draw_train_before(i)
            self.window.flip()
            _ = event.waitKeys(keyList=['space'])

        color_design,lr_design = self.fixed_design(n_trial)

        designs = self.generate_fixed_designs()
        self.show_countdown()
        for trial, (_, design) in enumerate(designs.iterrows()):
            if trial >= n_trial:
                break
            _ = self.run_trial(design,color_design[trial],lr_design[trial])

        self.drawer.draw_train_after()
        self.window.flip()
        _ = event.waitKeys(keyList=['space'])

    def fixed_design(self, n_trial):
        #color and direction: balanced random
        half_trial = n_trial//2
        color_design = [0]*half_trial+[1]*(n_trial-half_trial) #color
        lr_design = [-1]*half_trial+[1]*(n_trial-half_trial)
        np.random.shuffle(color_design)
        np.random.shuffle(lr_design)

        return color_design,lr_design

    def run_block(self, block, block_type, n_trial=30):
        """Run a block with optimal designs chosen by ADO."""
        self.engine.reset()

        color_design,lr_design = self.fixed_design(n_trial)

        if block_type == 'fixed':
            designs = self.generate_fixed_designs()

        self.show_countdown()
        for trial in range(n_trial):
            if block_type == 'fixed':
                design = designs.iloc[trial, ]
                color = design['color']
            else:
                design = self.engine.get_design('optimal')
                color = color_design[trial]
            lr = lr_design[trial]

            resp, rt = self.run_trial(design,color,lr)
            self.engine.update(design, resp)

            dict_mean = {
                'mean_' + p: m
                for p, m in zip(self.model.params, self.engine.post_mean)
            }
            dict_sd = {
                'sd_' + p: m
                for p, m in zip(self.model.params, self.engine.post_sd)
            }

            self.df = pd.concat([self.df, pd.DataFrame([{
                'subject': self.subj,
                'block': block,
                'block_type': block_type,
                'trial': trial + 1,
                **design,
                'resp_var': resp,
                'rt': rt,
                'color': color,
                'lr': lr,
                **dict_mean,
                **dict_sd
            }])], ignore_index=True)

            self.save_record()

def run_cra_ado(subj, window):
    n_block = 1
    n_trial = 30
    n_train_trial = 5
    has_tutorial = True

    time_now = datetime.now()
    time_now_iso = time_now.isoformat().replace(':', '-').replace('T', '-')[:-7]
    
    # Save Data
    path_data = PATH_ROOT.parent.parent / 'data' / f"{subj}"
    fn_data = f"CRA_{time_now_iso}.csv"
    path_data.mkdir(exist_ok=True)
    path_output = str(path_data / fn_data)

    block_types = ['ado'] * n_block

    print('Block types:', block_types)

    # Initialize a drawer and a runner
    drawer = CraDrawer(window)
    runner = CraRunner(window, drawer, subj, path_output)

    # Run blocks
    if has_tutorial:
        runner.show_intro()
        runner.run_train_block(n_train_trial)

    for block, block_type in enumerate(block_types):
        runner.show_block_start(block + 1, n_trial)
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

    run_cra_ado(args.subjId, window)
