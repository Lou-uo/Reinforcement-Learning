import copy

""" 环境图示：
列:  0    1    2    3    4    5    6    7    8    9   10   11
   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
 0 │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9  │ 10 │ 11 │
   │    │    │    │    │    │    │    │    │    │    │    │    │
   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
 1 │ 12 │ 13 │ 14 │ 15 │ 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │
   │    │    │    │    │    │    │    │    │    │    │    │    │
   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
 2 │ 24 │ 25 │ 26 │ 27 │ 28 │ 29 │ 30 │ 31 │ 32 │ 33 │ 34 │ 35 │
   │    │    │    │    │    │    │    │    │    │    │    │    │
   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
 3 │ 36 │ 37 │ 38 │ 39 │ 40 │ 41 │ 42 │ 43 │ 44 │ 45 │ 46 │ 47 │
   │ S  │****│****│****│****│****│****│****│****│****│****│ G  │
   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
   
   S = Start (起点)    **** = Cliff (悬崖, 奖励-100)    G = Goal (目标)
"""

class CliffWalkingEnv:
    """ 悬崖漫步环境 """
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol    # 定义网格世界的列
        self.nrow = nrow    # 定义网格世界的行
        # 转移矩阵
        # P[state][action] = [(p, next_state, reward, done)]
        # [(概率,下一个状态,奖励,是否终止)]
        self.P = self.createP()

    def createP(self):
        # 初始化 P[状态][动作] = 转移列表
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作，change[0]:上,change[1]:下,change[2]:左,change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):      # 遍历行
            for j in range(self.ncol):  # 遍历列
                for a in range(4):      # 遍历4个动作
                    # 位置在悬崖或者目标状态，因为无法继续交互，任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 其他位置（边界检查）
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1  # 默认每步奖励-1
                    done = False  # 默认未终止
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

""" 算法核心：
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  随机初始化  │ ──→ │  策略评估    │ ──→ │  策略提升   │
│   策略 π    │      │ 计算 V^π    │     │ 得到更好π'  │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
        ↑                  │                    │
        └──────────────────┘←───────────────────┘
                    若 π' = π，停止
"""

class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env # 环境对象
        self.v = [0] * self.env.ncol * self.env.nrow # 状态价值函数，初始全0
        self.pi = [[0.25, 0.25, 0.25, 0.25]
            for i in range(self.env.nrow * self.env.ncol)] # 均匀随机策略
        self.theta = theta # 收敛阈值 (如0.0001)
        self.gamma = gamma # 折扣因子 (如0.9)

    def policy_evaluation(self): # 策略评估
        cnt = 1 # 初始化计数器
        while True: # 无限循环，直到收敛
            max_diff = 0 # 记录本轮最大更新幅度
            new_v = [0] * self.env.nrow * self.env.ncol # 新一轮的价值估计
            for s in range(self.env.nrow * self.env.ncol): # 遍历所有状态
                qsa_list = [] # 开始计算状态 s下的所有 Q(s,a)价值
                for a in range(4): # 遍历4个动作
                    qsa = 0
                    for res in self.env.P[s][a]: # 遍历所有可能转移
                        p, next_state, r, done = res

                         # 核心公式：Q(s,a) = Σ p(s',r|s,a) * [r + γ * V(s') * (1-done)]
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                        # 本章环境比较特殊，奖励和下一个状态有关，所有需要和状态转移概率相乘
                    qsa_list.append(self.pi[s][a] * qsa) # π(a|s) * Q(s,a)
                new_v[s] = sum(qsa_list) # 状态价值函数和动作价值函数之间的关系 V(s) = Σ π(a|s) * Q(s,a)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v # 更新价值函数
            if max_diff < self.theta: break # 满足收敛条件(最大更新幅度 < 阈值)，退出评估迭代
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self): # 策略提升
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa) # 纯 Q值，不加权
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq) # 计算有几个动作得到了最大的 Q值
            # 让这些动作均分概率，其他动作概率为0
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):
        while True:
            self.policy_evaluation()            # 步骤1：评估当前策略
            old_pi = copy.deepcopy(self.pi)     # 深拷贝保存旧策略,方便接下来进行比较
            new_pi = self.policy_improvement()  # 步骤2：提升策略
            if old_pi == new_pi: break          # 策略不再变化，收敛！
    """ 为什么使用 deepcopy?
        错误: old_pi = self.pi  ← 只是引用，指向同一对象
        正确: old_pi = copy.deepcopy(self.pi)  ← 完全独立的副本

        因为 policy_improvement 会修改 self.pi 的内容
        不用深拷贝则 old_pi 也会被修改，无法比较
    """

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print("****", end=' ')
            elif (i * agent.env.ncol + j) in end:
                print("EEEE", end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])

""" 运行结果：
策略评估进行60轮后完成
策略提升完成
策略评估进行72轮后完成
策略提升完成
策略评估进行44轮后完成
策略提升完成
策略评估进行12轮后完成
策略提升完成
策略评估进行1轮后完成
策略提升完成
状态价值:
-7.712 -7.458 -7.176 -6.862 -6.513 -6.126 -5.695 -5.217 -4.686 -4.095 -3.439 -2.710
-7.458 -7.176 -6.862 -6.513 -6.126 -5.695 -5.217 -4.686 -4.095 -3.439 -2.710 -1.900
-7.176 -6.862 -6.513 -6.126 -5.695 -5.217 -4.686 -4.095 -3.439 -2.710 -1.900 -1.000
-7.458  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
策略:
ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovoo
ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovo> ovoo
ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ooo> ovoo
^ooo **** **** **** **** **** **** **** **** **** **** EEEE
"""

""" 下面是OpenAI Gym 库的冰湖环境模拟
import gym
env = gym.make("FrozenLake-v0")  # 创建环境
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)

for a in env.P[14]:  # 查看目标左边一格的状态转移信息
    print(env.P[14][a])

# 这个动作意义是Gym库针对冰湖环境事先规定好的
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])
"""