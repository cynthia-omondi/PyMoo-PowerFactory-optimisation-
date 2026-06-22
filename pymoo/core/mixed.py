import math
import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.infill import InfillCriterion
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Choice, Real, Integer, Binary
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.core.repair import Repair
from pymoo.util.display.multi import MultiObjectiveOutput
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.selection.tournament import compare, TournamentSelection

from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible


def binary_tournament(pop, P, algorithm, random_state=None, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = tournament_type = 'comp_by_rank_and_crowding' # hard coded
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True, random_state=algorithm.random_state)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision, compare by crowding
            if np.isnan(S[i]):
                # Guard against None crowding distances (e.g. first generation)
                if cd_a is None or cd_b is None:
                    S[i] = a if random_state.random() < 0.5 else b
                else:
                    S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better',
                                   return_random_if_equal=True, random_state=algorithm.random_state)


    return S[:, None].astype(int, copy=False)
    
class StructuredCrossover(Crossover):
    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    @staticmethod
    def _build_segments(var_names: list):
        # group 1: EV / RES / IND prefix groups
        ev_idx  = [i for i, n in enumerate(var_names) if n.startswith("EV_")]
        ind_idx = [i for i, n in enumerate(var_names) if n.startswith("IND_")]
        res_idx = [i for i, n in enumerate(var_names) if n.startswith("RES_")]
        
        prefix_groups = [g for g in [ev_idx, ind_idx, res_idx, ] if g]

        # group 2: xab/yab pairs
        seen  = set()
        pairs = []
        for name in var_names:
            if name.startswith("xab_"):
                suffix  = name[len("xab_"):]
                yab_key = f"yab_{suffix}"
                if suffix not in seen and yab_key in var_names:
                    xi = var_names.index(name)
                    yi = var_names.index(yab_key)
                    pairs.append((xi, yi))
                    seen.add(suffix)

        # group 3: xas/sas/cas/das triplets
        seen_t   = set()
        triplets = []
        for name in var_names:
            if name.startswith("xas_"):
                suffix  = name[len("xas_"):]
                sas_key = f"sas_{suffix}"
                cas_key = f"cas_{suffix}"
                das_key = f"das_{suffix}"
                yas_key = f"yas_{suffix}"
                if suffix not in seen_t:
                    xi      = var_names.index(name)
                    si      = var_names.index(sas_key) if sas_key in var_names else None
                    ci      = var_names.index(cas_key) if cas_key in var_names else None
                    di      = var_names.index(das_key) if das_key in var_names else None
                    yi1      = var_names.index(yas_key) if das_key in var_names else None
                    # store as list of only existing indices
                    members = [idx for idx in [xi, si, ci, di, yi1] if idx is not None]
                    triplets.append(members)
                    seen_t.add(suffix)

        return prefix_groups, pairs, triplets

    def _do(self, problem, X, random_state=None, **kwargs):
        _, n_matings, n_var = X.shape

        var_names = list(problem.vars.keys())

        prefix_groups, pairs, triplets = self._build_segments(var_names)

        Xp = X.copy()

        for m in range(n_matings):
            a, b = X[0, m], X[1, m]
            o1, o2 = a.copy(), b.copy()

            # ── group 1: 2-point crossover on prefix groups as atoms ──────────
            if len(prefix_groups) > 1:
                self._n_point_atomic(o1, o2, a, b, prefix_groups, n_points=2,
                                     random_state=random_state)
            #group 2
            if len(pairs) > 1:
                self._n_point_atomic(o1, o2, a, b, pairs, n_points=len(pairs)-1,
                                     random_state=random_state)

            # ── group 3: 2-point crossover on xas triplets as atoms ─────────
            if len(triplets) > 1:
                self._n_point_atomic(o1, o2, a, b, triplets, n_points=len(triplets)-1,
                                     random_state=random_state)

            Xp[0, m] = o1
            Xp[1, m] = o2

        return Xp

    @staticmethod
    def _n_point_atomic(o1, o2, a, b, atoms, n_points, random_state):
        n = len(atoms)
        if n < 2:
            return
        n_points = min(n_points, n - 1)
        cuts = sorted(
            random_state.choice(np.arange(1, n), size=n_points, replace=False)
        )
        cuts.append(n)
        swap = False
        prev = 0
        for cut in cuts:
            if swap:
                for ai in range(prev, cut):
                    for idx in atoms[ai]:
                        o1[idx], o2[idx] = b[idx], a[idx]
            swap = not swap
            prev = cut


class StructuredMutation(Mutation):
    def __init__(self, prob=1.0, prob_var=0.1, sigma_frac=0.1, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.prob_var   = prob_var
        self.sigma_frac = sigma_frac
    @staticmethod
    
    def _build_segments(var_names: list):
        ev_idx  = [i for i, n in enumerate(var_names) if n.startswith("EV_")]
        res_idx = [i for i, n in enumerate(var_names) if n.startswith("RES_")]
        ind_idx = [i for i, n in enumerate(var_names) if n.startswith("IND_")]
        prefix_groups = [g for g in [ev_idx, ind_idx, res_idx] if g]
    
        seen  = set()
        pairs = []
        for name in var_names:
            if name.startswith("xab_"):
                suffix  = name[len("xab_"):]
                yab_key = f"yab_{suffix}"
                if suffix not in seen and yab_key in var_names:
                    xi = var_names.index(name)
                    yi = var_names.index(yab_key)
                    pairs.append((xi, yi))
                    seen.add(suffix)
    
        seen_t   = set()
        triplets = []
        for name in var_names:
            if name.startswith("xas_"):
                suffix  = name[len("xas_"):]
                sas_key = f"sas_{suffix}"
                cas_key = f"cas_{suffix}"
                das_key = f"das_{suffix}"
                yas_key = f"yas_{suffix}"
                if suffix not in seen_t:
                    xi  = var_names.index(name)
                    si  = var_names.index(sas_key) if sas_key in var_names else None
                    ci  = var_names.index(cas_key) if cas_key in var_names else None
                    di  = var_names.index(das_key) if das_key in var_names else None
                    yi1 = var_names.index(yas_key) if yas_key in var_names else None  # FIX 2
                    triplets.append({
                        "xas": xi,
                        "sas": si,
                        "cas": ci,
                        "das": di,
                        "yas": yi1,
                    })
                    seen_t.add(suffix)
    
        return prefix_groups, pairs, triplets

    def _mutate_float(self, value, name, problem, random_state):
        var = problem.vars[name]
        lo, hi = var.bounds
        sigma = (hi - lo) * self.sigma_frac
        new_val = value + random_state.normal(0, sigma)
        return float(np.clip(new_val, lo, hi))
    
    def _mutate_int(self, value, name, problem, random_state):
        var = problem.vars[name]
        lo, hi = var.bounds
        sigma = max(1, (hi - lo) * self.sigma_frac)
        new_val = value + random_state.normal(0, sigma)
        return int(np.clip(round(new_val), lo, hi))
        
    def _mutate_choice(self, value, name, problem, random_state):
        var = problem.vars[name]
        options = var.options  # or var.choices — depends on your Variable class attribute name
        
        if len(options) <= 1:
            return value  # nothing to mutate to
        
        # Pick from all options except current value
        other_options = [o for o in options if o != value]
        return random_state.choice(other_options)
        
    def _do(self, problem, X, random_state=None, **kwargs):
        var_names = list(problem.vars.keys())
        prefix_groups, pairs, triplets = self._build_segments(var_names)
    
        mutated_count = 0
    
        for i in range(len(X)):
            if random_state.random() > float(self.prob_var):
                continue
    
            ind = X[i].X.copy()   # <-- .X to get the dict
    
            # ── group 1: EV_ / RES_ / IND_ floats ────────────────────
            for group_idx in prefix_groups:
                for idx in group_idx:
                    name = var_names[idx]
                    ind[name] = self._mutate_float(ind[name], name, problem, random_state)
                    mutated_count += 1
    
            # ── group 2: xab_/yab_ pairs ─────────────────────────────
            for xi, yi in pairs:
                xab_name = var_names[xi]
                yab_name = var_names[yi]
                if ind[xab_name] == 1:
                    ind[yab_name] = self._mutate_int(ind[yab_name], yab_name, problem, random_state)
                    mutated_count += 1
    
            # ── group 3: xas_ triplets ────────────────────────────────
            for members in triplets:
                xi  = members["xas"]
                si  = members["sas"]
                ci  = members["cas"]
                di  = members["das"]
                yi1 = members["yas"]
    
                xas_name = var_names[xi]
                if ind[xas_name] == 1:
                    for idx, mutate_fn in [
                        (si,  self._mutate_choice),
                        (ci,  self._mutate_choice),
                        (di,  self._mutate_choice),
                        (yi1, self._mutate_int),
                    ]:
                        if idx is None:
                            continue
                        vname = var_names[idx]
                        ind[vname] = mutate_fn(ind[vname], vname, problem, random_state)
                        mutated_count += 1
    
            X[i].X = ind   # write back

        return X
      
class MixedVariableMating(InfillCriterion):

    def __init__(self,
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=StructuredCrossover(),
                 mutation=StructuredMutation(),
                 repair=None,
                 eliminate_duplicates=True,
                 n_max_iterations=100,
                 **kwargs):

        super().__init__(repair, eliminate_duplicates, n_max_iterations, **kwargs)
        

        if mutation is None:
            mutation = {
                Binary:  BFM(),
                Real:    PM(),
                Integer: PM(vtype=float, repair=RoundingRepair()),
                Choice:  ChoiceRandomMutation(),
            }
                 
        self.selection = selection

        self.crossover = crossover

        self.mutation  = mutation   


    def _do(self, problem, pop, n_offsprings, parents=False, random_state=None, **kwargs):
    
        XOVER_N_PARENTS    = 2
        XOVER_N_OFFSPRINGS = 2
    
        # ── selection ─────────────────────────────────────────────────
        if not parents:
            n_select = math.ceil(n_offsprings / XOVER_N_OFFSPRINGS)
            pop = self.selection(
                problem, pop, n_select, XOVER_N_PARENTS,
                random_state=random_state, **kwargs
            )
    
        # ── build parent array: shape (2, n_matings, n_var) ───────────
        n_matings = len(pop)
        var_names = list(problem.vars.keys())
        n_var     = len(var_names)
    
        X = np.empty((XOVER_N_PARENTS, n_matings, n_var), dtype=object)
        for m, pair in enumerate(pop):
            for p, ind in enumerate(pair):
                for j, name in enumerate(var_names):
                    X[p, m, j] = ind.X[name]
    
        # ── crossover → shape (2, n_matings, n_var) ───────────────────
        Xp = self.crossover._do(problem, X, random_state=random_state, **kwargs)
    
        # ── flatten crossover output to array of dicts ────────────────
        n_off = XOVER_N_OFFSPRINGS * n_matings
        X_mut = np.empty(n_off, dtype=object)
        for i in range(n_matings):
            for o in range(XOVER_N_OFFSPRINGS):
                X_mut[o * n_matings + i] = {
                    name: Xp[o, i, j] for j, name in enumerate(var_names)
                }
    
        _off = Population.new(X=list(X_mut))
        _off = self.mutation._do(problem, _off, random_state=random_state, **kwargs)
    
        # ── pack into Population, trim to n_offsprings ────────────────
        off = Population.new(X=[ind.X for ind in _off[:n_offsprings]])
    
        return off


class MixedVariableSampling(Sampling):

    def _do(self, problem, n_samples, random_state=None, **kwargs):
        V = {name: var.sample(n_samples, random_state=random_state) for name, var in problem.vars.items()}

        X = []
        for k in range(n_samples):
            X.append({name: V[name][k] for name in problem.vars.keys()})

       # # Initial population is generated here
        return X


class MixedVariableDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        a, b = a.X, b.X
        for k, v in a.items():
            if k not in b or b[k] != v:
                return False
        return True


def groups_of_vars(vars):
    ret = {}
    for name, var in vars.items():
        if var.__class__ not in ret:
            ret[var.__class__] = []

        ret[var.__class__].append((name, var))

    return ret


class MixedVariableGA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=50,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 sampling=MixedVariableSampling(),
                 mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                 eliminate_duplicates=MixedVariableDuplicateElimination(),
                 survival=FitnessSurvival(),
                 **kwargs):
        super().__init__(pop_size=pop_size, n_offsprings=n_offsprings, sampling=sampling, mating=mating,
                         eliminate_duplicates=eliminate_duplicates, output=output, survival=survival, **kwargs)
