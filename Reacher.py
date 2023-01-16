import os.path
import gurobipy as gp
from gurobipy import GRB
import scipy
import numpy
import gym
import mujoco_py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import string
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # make render not lag
os.environ['TF_DETERMINISTIC_OPS'] = '1'
ewma_r = 0  # ewma for reward
arg_seed = 0
######################### seed ##############################
#random.seed(arg_seed)
#np.random.seed(arg_seed)
tf.compat.v1.reset_default_graph()
##################### hyper parameters ####################
LR_C = 0.001    # learning rate for critic
LR_A = 0.0001   # learning rate for actor
GAMMA = 0.99    # determine the importance of future reward
TAU = 0.001

MEMORY_CAPACITY = 10000 # the size the replay buffer
BATCH_SIZE = 64
eval_freq = 5000
store_testing_before_action = []
store_testing_after_action = []

#################### testing part #################################
def evaluation(env_name, seed, ddpg, c, eval_episode=10):
    avgreward = 0
    avg = []
    eval_env = gym.make(env_name)
    eval_env.seed(seed+100)
    for eptest in range(eval_episode):
        running_reward = 0
        done = False
        s = eval_env.reset()
        while not done:
            action = ddpg.choose_action(s)
            store_testing_before_action.append(action)
            action, loss = Projection(action, c) # add the parameter
            store_testing_after_action.append(action)
            s_, r, done, info = eval_env.step(action)
            s = s_
            running_reward = running_reward+r
        print('Episode {}\tReward: {} \t AvgReward'.format(eptest, running_reward))
        avgreward = avgreward+running_reward
        avg.append(running_reward)
    avgreward = avgreward/eval_episode
    print("------------------------------------------------")
    print("Evaluation average reward :", avgreward)
    print("------------------------------------------------")

    return avgreward

###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, env):
        
        # initial envirnment here
        self.env = env
        self.s_dim = self.env.observation_space.shape[0]  # dim of observation space: 11
        self.a_dim = self.env.action_space.shape[0]   # dim of action space: 2
        self.a_bound = self.env.action_space.high # bound of action value: (1, 1)

        self.memory = np.zeros(
            (MEMORY_CAPACITY, self.s_dim * 2 + self.a_dim + 1+1), dtype=np.float32)
        self.pointer = 0    # point to the index in replay buffer
        configuration = tf.compat.v1.ConfigProto()  # get configuration
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)  # use the configuration to approve session
        tf.random.set_seed(arg_seed)

        self.S = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], 's')   # current state
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], 's_') # next state
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')   # reward
        self.Done = tf.compat.v1.placeholder(tf.float32, [None, 1], 'done') # if the task is done

        # create the unique name in order not to get the same scope
        self.length_of_name = 5
        self.actor_name = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(self.length_of_name))
        self.critic_name = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(self.length_of_name))

        with tf.compat.v1.variable_scope(self.actor_name):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope(self.critic_name):
            self.q = self._build_c(
                self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.actor_name+"/eval")
        self.at_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.actor_name+"/target")
        self.ce_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.critic_name+"/eval")
        self.ct_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.critic_name+"/target")

        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + (1-self.Done)*GAMMA * q_
        td_error = tf.compat.v1.losses.mean_squared_error(
            labels=q_target, predictions=self.q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(
            LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(input_tensor=self.q)    # maximize the q

        self.atrain = tf.compat.v1.train.AdamOptimizer(
            LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):

        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        buffer_size = min(self.pointer+1, MEMORY_CAPACITY)
        indices = np.random.choice(buffer_size, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim+self.a_dim:self.s_dim+self.a_dim+1]
        bs_ = bt[:, self.s_dim+self.a_dim+1:self.s_dim+self.a_dim+1+self.s_dim]
        bd = bt[:, -1:]
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(
            self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.Done: bd})

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, done))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(
                s, 400, activation=tf.nn.relu, name='l1', trainable=trainable)  # a hidden layers with dim 400
            net2 = tf.compat.v1.layers.dense(
                net, 300, activation=tf.nn.relu, name='l2', trainable=trainable)    # a hidden layers with dim 300
            a = tf.compat.v1.layers.dense(
                net2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable) # the last layer that output result
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(
                s, 400, activation=tf.nn.relu, name='cl1', trainable=trainable)
            w2_net = tf.compat.v1.get_variable(
                'w2_net', [400, 300], trainable=trainable)
            w2_a = tf.compat.v1.get_variable(
                'w2_a', [self.a_dim, 300], trainable=trainable)
            b2 = tf.compat.v1.get_variable('b1', [1, 300], trainable=trainable)
            net2 = tf.nn.relu(tf.matmul(a, w2_a)+tf.matmul(net, w2_net)+b2)
            # Q(s,a)
            return tf.compat.v1.layers.dense(net2, 1, trainable=trainable)
    
    def get_gradient(self, s, a):
        return self.sess.run(self.action_grad, {self.S: s, self.a: a})

    def fw_update(self):
        lr = 0.05
        buffer_size = min(ddpg.pointer+1, MEMORY_CAPACITY)
        indices = np.random.choice(buffer_size, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = self.sess.run(self.a, {self.S: bs})
        #######let ba in constraint######
        for i in range(BATCH_SIZE):
            ba[i] = Projection(ba[i])

        #######let ba in constraint######
        grad = self.get_gradient(bs, ba)
        grad = np.squeeze(grad)
        action_table = np.zeros((BATCH_SIZE, 2))  # (16,16,2
        for i in range(BATCH_SIZE):
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model(env=env) as fw_m:
                    a1 = fw_m.addVar(lb=-1, ub=1, name="a1",
                                     vtype=GRB.CONTINUOUS)
                    a2 = fw_m.addVar(lb=-1, ub=1, name="a2",
                                     vtype=GRB.CONTINUOUS)
                    obj = a1*grad[i][0]+a2*grad[i][1]
                    fw_m.setObjective(obj, GRB.MAXIMIZE)
                    # fw_m.addConstr(a1+a2 <= 0.1)
                    # fw_m.addConstr(-0.1 <= a1+a2)
                    fw_m.addConstr((a1**2+a2**2) <= 0.05)
                    fw_m.optimize()
                    action_table[i][0] = a1.X
                    action_table[i][1] = a2.X
                action_table[i] = action_table[i]*lr+ba[i] * (1-lr)
        self.sess.run(self.update, {self.S: bs, self.tf_table: action_table})

#############################  optlayer  ####################################
def Projection(action, c):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as reacher_m:
            neta1 = action[0]
            neta2 = action[1]

            a1 = reacher_m.addVar(lb=-1, ub=1, name="a1", vtype=GRB.CONTINUOUS)
            a2 = reacher_m.addVar(lb=-1, ub=1, name="a2", vtype=GRB.CONTINUOUS)
            obj = (a1-neta1)**2 + (a2-neta2)**2
            reacher_m.setObjective(obj, GRB.MINIMIZE)

            # reacher_m.addConstr(a1+a2 <= 0.1)
            # reacher_m.addConstr(-0.1 <= a1+a2)
            reacher_m.addConstr((a1**2/c[0]**2+a2**2/c[1]**2) <= 1)

            reacher_m.optimize()

            return (a1.X, a2.X), reacher_m.objVal

# contraints
con_pool = []  # create a ddpg_pool of tasks
len_pool = 10
for i in range(len_pool):
    c = []
    c.append(random.uniform(0, 1))
    c.append(random.uniform(0, 1))
    con_pool.append(c)

env_name = 'Reacher-v2'

batch_times = 10    # times of sampling batches
for _ in range(batch_times):
    # sample a batch
    con_batch_size = 3 # define num of tasks in a batch
    con_batch  = random.sample(con_pool, con_batch_size)
    
    for c in con_batch:
        tmp_env = gym.make(env_name)
        ddpg = DDPG(tmp_env)
        ewma = []
        eva_reward = []
        store_before_action = []
        store_after_action = []
        store_before_action_and_gaussain = []
        reward = []
        nploss = []
        i = 0
        Net_action = np.zeros((100000, ddpg.a_dim+2))
        max_action = float(ddpg.env.action_space.high[0])
        step = 0
        for ep in range(10): # num of episode(I modified it for lack of cpu resource)
            # env.render()
            step = 0    
            R = 0
            done = False
            s = ddpg.env.reset()
            while not done:
                step = step+1
                if ddpg.pointer < 1000: # if the ddqn hasn't done 1000 transition
                    action = ddpg.env.action_space.sample()  # sample a set of action from action space
                else:
                    action = ddpg.choose_action(s)  # if the machine has run over 1000 transition, let ddqg choose the action
                    store_before_action_and_gaussain.append(action) # don't know what it's for, it's saved at the end
                    action = (action+np.random.normal(0, 0.1, ddpg.a_dim)
                            ).clip(-max_action, max_action)   # adding noise to the action maybe? this is the gaussion thingy
                    i = i+1
                store_before_action.append(action)  # don't know what it's for either, saved at the end
                action, loss = Projection(action, c)   # the new action is the projected action
                store_after_action.append(action)
                # assert abs(sum(action)) <= 0.1+1e-6
                # assert -1 <= action[0] <= 1
                # assert -1 <= action[1] <= 1
                assert abs((action[0]**2+action[1]**2)) <= 1    

                s_, r, done, info = ddpg.env.step(action)    # after appling action, return state, reward, info(not used)

                done_bool = False if step == ddpg.env._max_episode_steps else done  # unclear it seems if step ==  max_episode_steps, the done_bool is 
                                                                            # set to false even if the done is true, but the while loop is
                                                                            # controled by done?

                ddpg.store_transition(s, action, r, s_, done_bool)  #store the transition
                if ddpg.pointer >= 1000:    
                    ddpg.learn()
                if (ddpg.pointer+1) % eval_freq == 0:
                    eva_reward.append(evaluation(env_name, arg_seed, ddpg, c))
                s = s_
                R += r

            ewma_r = 0.05 * R + (1 - 0.05) * ewma_r
            print({
                'episode': ep,
                'reward': R,
                'ewma_reward': ewma_r # smooth the reward
            })

            reward.append(R)
            ewma.append(ewma_r)
            if(ddpg.pointer >= 500000):
                print("done training")
                break

np.save("Reacher_{}_Projection_Reward".format(arg_seed), reward)
np.save("Reacher_{}_Projection_before_Action".format(
    arg_seed), store_before_action)
np.save("Reacher_{}_Projection_after_Action".format(
    arg_seed), store_after_action)
np.save("Reacher_{}_Projection_eval_reward".format(arg_seed), eva_reward)
np.save("Reacher_{}_Projection_before_Action_Gaussian".format(
    arg_seed), store_before_action_and_gaussain)
np.save("Reacher_{}_Projection_eval_before_Action".format(
    arg_seed), store_testing_before_action)
np.save("Reacher_{}_Projection_eval_After_Action".format(
    arg_seed), store_testing_after_action)