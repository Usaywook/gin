from algos.grain import GRAIN
from algos.gin import GIN, Ably
from algos.sac import SAC
from algos.wcsac import WCSAC
from algos.cpo import CPO
from algos.trpo import TRPO
from algos.grip import GRIP, VanilaGRU
from algos.cgrain import CGRAIN
from utils import prRed

def get_agent(obs_space, act_space, args):
    if args.algo == 'grain':
        agent = GRAIN(obs_space, act_space, args)
    elif args.algo == 'sac':
        agent = SAC(obs_space, act_space, args)
    elif args.algo == 'wcsac':
        agent = WCSAC(obs_space, act_space, args)
    elif args.algo == 'gin':
        if args.study == 'temporal' or args.study == 'social' or args.study == 'hmm' or args.study == 'random':
            agent = Ably(obs_space, act_space, args)
        else:
            agent = GIN(obs_space, act_space, args)
    elif args.algo == 'cpo':
        agent = CPO(obs_space, act_space, args)
    elif args.algo == 'cgrain':
        agent = CGRAIN(obs_space, act_space, args)
    elif args.algo == 'trpo':
        agent = TRPO(obs_space, act_space, args)
    elif args.algo == 'grip':
        if args.study == 'v-gru':
            agent = VanilaGRU(obs_space, act_space, args)
        else:
            agent = GRIP(obs_space, act_space, args)
    else:
        prRed("choose proper algo : grain | sac | wcsac | cpo | cgrain | grip")

    if args.cuda:   agent.cuda()
    if args.write_summary:  agent.set_writer(args)

    return agent