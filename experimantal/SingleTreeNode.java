package players.experimantal;

import core.GameState;
import players.heuristics.AdvancedHeuristic;
import players.heuristics.CustomHeuristic;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;
import utils.Types;
import utils.Utils;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

public class SingleTreeNode
{
    public MCTSParamsAK params;

    private SingleTreeNode parent;
    private SingleTreeNode[] children;
    private double totValue;
    private int nVisits;
    private Random m_rnd;
    private int m_depth;
    private double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    private int childIdx;
    private int fmCallsCount;
    private double stateHeursitic;
    private int num_actions;
    private Types.ACTIONS[] actions;

    private GameState rootState;
    private StateHeuristic rootStateHeuristic;
    double discReward;
    double gamma = 0.3;
    public double epsilon = 1e-6;

    SingleTreeNode(MCTSParamsAK p, Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this(p, null, -1, rnd, num_actions, actions, 0, null);
    }

    private SingleTreeNode(MCTSParamsAK p, SingleTreeNode parent, int childIdx, Random rnd, int num_actions,
                           Types.ACTIONS[] actions, int fmCallsCount, StateHeuristic sh) {
        this.params = p;
        this.fmCallsCount = fmCallsCount;
        this.parent = parent;
        this.m_rnd = rnd;
        this.num_actions = num_actions;
        this.actions = actions;
        children = new SingleTreeNode[num_actions];
        totValue = 0.0;
        this.childIdx = childIdx;
        if(parent != null) {
            m_depth = parent.m_depth + 1;
            this.rootStateHeuristic = sh;
        }
        else
            m_depth = 0;
    }

    void setRootGameState(GameState gs)
    {
        this.rootState = gs;
        if (params.heuristic_method == params.CUSTOM_HEURISTIC)
            this.rootStateHeuristic = new CustomHeuristic(gs);
        else if (params.heuristic_method == params.ADVANCED_HEURISTIC) // New method: combined heuristics
            this.rootStateHeuristic = new AdvancedHeuristic(gs, m_rnd);
    }


    void mctsSearch(ElapsedCpuTimer elapsedTimer) {

        double avgTimeTaken;
        double acumTimeTaken = 0;
        long remaining;
        int numIters = 0;

        int remainingLimit = 5;
        boolean stop = false;

        while(!stop){

            GameState state = rootState.copy();
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            SingleTreeNode selected = treePolicy(state);
            double delta = selected.rollOut(state);
            backUp(selected, delta);

            //Stopping condition
            if(params.stop_type == params.STOP_TIME) {
                numIters++;
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
                avgTimeTaken  = acumTimeTaken/numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            }else if(params.stop_type == params.STOP_ITERATIONS) {
                numIters++;
                stop = numIters >= params.num_iterations;
            }else if(params.stop_type == params.STOP_FMCALLS)
            {
                fmCallsCount+=params.rollout_depth;
                stop = (fmCallsCount + params.rollout_depth) > params.num_fmcalls;
            }
        }
        //System.out.println(" ITERS " + numIters);
    }

    private SingleTreeNode treePolicy(GameState state) {

        SingleTreeNode cur = this;

        while (!state.isTerminal() && cur.m_depth < params.rollout_depth)
        {
            if (cur.notFullyExpanded()) {
                return cur.expand(state);

            } else {
                cur = cur.uct(state);
            }
        }

        return cur;
    }


    private SingleTreeNode expand(GameState state) {

        int bestAction = 0;
        double bestValue = -1;

        for (int i = 0; i < children.length; i++) {
            double x = m_rnd.nextDouble();
            if (x > bestValue && children[i] == null) {
                bestAction = i;
                bestValue = x;
            }
        }

        //Roll the state
        roll(state, actions[bestAction]);

        SingleTreeNode tn = new SingleTreeNode(params,this,bestAction,this.m_rnd,num_actions,
                actions, fmCallsCount, rootStateHeuristic);
        children[bestAction] = tn;
        return tn;
    }

    private void roll(GameState gs, Types.ACTIONS act)
    {
        //Simple, all random first, then my position.
        int nPlayers = 4;
        Types.ACTIONS[] actionsAll = new Types.ACTIONS[4];
        int playerId = gs.getPlayerId() - Types.TILETYPE.AGENT0.getKey();

        for(int i = 0; i < nPlayers; ++i)
        {
            if(playerId == i)
            {
                actionsAll[i] = act;
            }else {
                int actionIdx = m_rnd.nextInt(gs.nActions());
                actionsAll[i] = Types.ACTIONS.all().get(actionIdx);
            }
        }

        gs.next(actionsAll);

    }

    private SingleTreeNode uct(GameState state) {
        SingleTreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;
        for (SingleTreeNode child : this.children)
        {
            double hvVal = child.totValue;
            double childValue =  hvVal / (child.nVisits + params.epsilon);

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);

            int nEnemies = state.getAliveEnemyIDs().size();
            double hSA = params.ADVANCED_HEURISTIC/( 1 + (child.nVisits + params.epsilon));
            double uctValue = childValue +
                   params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon)) +hSA;
//            double uctValue;
//            if (nEnemies == 3){
//                uctValue = childValue +
//                        (params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon))) + hSA;// favour exploring
//            } else if (nEnemies == 2){
//                uctValue = childValue +
//                        (params.K2 * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon))) + hSA ; //progressive bias and conservative
//            }else{
//                uctValue = childValue +
//                        (params.K3 * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon))); //favour exploiting with progressive bias
//            }

            uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > bestValue) {
                selected = child;
                bestValue = uctValue;
            }
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]);

        return selected;
    }

    private double rollOut(GameState state)
    {
        int thisDepth = this.m_depth;

        while (!finishRollout(state,thisDepth)) {
            int action = safeRandomAction(state);
            roll(state, actions[action]);
            thisDepth++;
        }
        //System.out.println(rootStateHeuristic.evaluateState(state));
//        stateHeursitic = rootStateHeuristic.evaluateState(state);
//        int nEnemies2 = state.getAliveEnemyIDs().size();
//        if (nEnemies2 >= 2){
//            discReward = stateHeursitic; // if enemies greater than 2 then explore and pick best action
//        }else if (nEnemies2 < 2){
//            discReward = stateHeursitic * Math.pow(gamma,thisDepth); //if 1 enemy left decay rollout and bias node selection closer nodes may be safer to play
//        }
//       return discReward;
        return rootStateHeuristic.evaluateState(state);
    }
    private int safeRandomAction(GameState state)
    {
        Types.TILETYPE[][] board = state.getBoard();
        ArrayList<Types.ACTIONS> actionsToTry = Types.ACTIONS.all();
        int width = board.length;
        int height = board[0].length;
        double maxQ = Double.NEGATIVE_INFINITY;
        int chosenAction = 0;
        int prevAction = chosenAction ;
        int stateTickSize = state.getTick();// returns number of alive enemies
        while(actionsToTry.size() > 0) {
            if (stateTickSize > 50) {
                int nAction = m_rnd.nextInt(actionsToTry.size());
                int rndSafeAction = m_rnd.nextInt(actionsToTry.size());

                Types.ACTIONS act = actionsToTry.get(nAction);
                Types.ACTIONS act2 = actionsToTry.get(prevAction);
                Types.ACTIONS act3 = actionsToTry.get(rndSafeAction);

                GameState gsCopy = state.copy();
                GameState gsCopy2 = state.copy();

                roll(gsCopy, act);
                roll(gsCopy2, act2);

                double valState = rootStateHeuristic.evaluateState(gsCopy);
                double prevValState = rootStateHeuristic.evaluateState(gsCopy2);
                double Q = Utils.noise(valState, this.epsilon, this.m_rnd.nextDouble());
                double Qprev = Utils.noise(prevValState, this.epsilon, this.m_rnd.nextDouble());


                if (Q > maxQ) {
                    maxQ = Q;
                    chosenAction = nAction;


                } else if (Qprev > maxQ && maxQ > 0) {
                    maxQ = Qprev;
                    chosenAction = prevAction;
                    act = act2;
                } else {
                    chosenAction = rndSafeAction;
                    act = act3;
                }

                Vector2d dir = act.getDirection().toVec();
                Vector2d pos = state.getPosition();
                int x = pos.x + dir.x;
                int y = pos.y + dir.y;

                if (x >= 0 && x < width && y >= 0 && y < height)
                    if (board[y][x] != Types.TILETYPE.FLAMES)
                        return chosenAction;


                actionsToTry.remove(nAction);
            } else {
                int nAction = m_rnd.nextInt(actionsToTry.size());
                Types.ACTIONS act = actionsToTry.get(nAction);
                Vector2d dir = act.getDirection().toVec();

                Vector2d pos = state.getPosition();
                int x = pos.x + dir.x;
                int y = pos.y + dir.y;

                if (x >= 0 && x < width && y >= 0 && y < height)
                    if(board[y][x] != Types.TILETYPE.FLAMES)
                        return nAction;
                    actionsToTry.remove(nAction);

            }
        }


        //Uh oh...
        return m_rnd.nextInt(num_actions);
    }

//    private Types.ACTIONS oslaAction(GameState state) {
//        rootStateHeuristic = new AdvancedHeuristic(state,m_rnd);
//        ArrayList<Types.ACTIONS> actionsList = Types.ACTIONS.all();
//        Types.TILETYPE[][] board = state.getBoard();
//
//        double maxQ = Double.NEGATIVE_INFINITY;
//        Types.ACTIONS bestAction = null;
//        Types.ACTIONS chosenAction = null;
//
//        for (Types.ACTIONS act : actionsList) {
//            GameState gsCopy = state.copy();
//            roll(gsCopy, act);
//            double valState = rootStateHeuristic.evaluateState(gsCopy);
//
//            double Q = Utils.noise(valState, this.epsilon, this.m_rnd.nextDouble());
//
//
//                      if (Q > maxQ) {
//                maxQ = Q;
//                bestAction = act;
//                chosenAction = bestAction;
//            }
//
//
//        }
//
//        return chosenAction;
//
//    }




    @SuppressWarnings("RedundantIfStatement")
    private boolean finishRollout(GameState rollerState, int depth)
    {
        if (depth >= params.rollout_depth)      //rollout end condition.
            return true;

        if (rollerState.isTerminal())               //end of game
            return true;

        return false;
    }

    private void backUp(SingleTreeNode node, double result)
    {
        SingleTreeNode n = node;
        while(n != null)
        {
            n.nVisits++;
            n.totValue += result;
            if (result < n.bounds[0]) {
                n.bounds[0] = result;
            }
            if (result > n.bounds[1]) {
                n.bounds[1] = result;
            }
            n = n.parent;
        }
    }


    int mostVisitedAction() {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;
        boolean allEqual = true;
        double first = -1;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null)
            {
                if(first == -1)
                    first = children[i].nVisits;
                else if(first != children[i].nVisits)
                {
                    allEqual = false;
                }

                double childValue = children[i].nVisits;
                childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            selected = 0;
        }else if(allEqual)
        {
            //If all are equal, we opt to choose for the one with the best Q.
            selected = bestAction();
        }

        return selected;
    }

    private int bestAction()
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null) {
                double childValue = children[i].totValue / (children[i].nVisits + params.epsilon);
                childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }


    private boolean notFullyExpanded() {
        for (SingleTreeNode tn : children) {
            if (tn == null) {
                return true;
            }
        }

        return false;
    }
}
