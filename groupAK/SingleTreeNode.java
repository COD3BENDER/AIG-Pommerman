package players.groupAK;

import core.GameState;

import players.heuristics.AdvancedHeuristic;
import players.heuristics.CustomHeuristic;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;
import utils.Types;
import utils.Utils;
import utils.Vector2d;

import java.util.*;

public class SingleTreeNode
{
    public MCTSParamsTD params;

    private SingleTreeNode parent;
    private SingleTreeNode[] children;
    private double totValue;
    private int nVisits;
    private Random m_rnd;
    private int m_depth;
    private double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    private int childIdx;
    private int fmCallsCount;
    private int num_actions;
    private Types.ACTIONS[] actions;
    private GameState rootState;
    private StateHeuristic rootStateHeuristic;
    private double gamma = 0.98; //used for the discount factor for decaying rewards
    private double rolloutType = 1; // choose 0 for default action selection 1 for OSLA action selection and 2 for Modified Action selection used in testing


    SingleTreeNode(MCTSParamsTD p, Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this(p, null, -1, rnd, num_actions, actions, 0, null);
    }

    private SingleTreeNode(MCTSParamsTD p, SingleTreeNode parent, int childIdx, Random rnd, int num_actions,
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
            double hSA = params.heuristic_method;// retrieve domain specific heuristic knowledge using the heuristic of the agent
            double nSA = (child.nVisits); // this was created to use in progressive bias

            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);


            double uctValue = childValue +
                    params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon)) + (hSA/(1+nSA)); //progressive bias - shows improvement using the formula h(s,a)/1 + N(s,a)
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

        if (rolloutType == 0) {
            int action = safeRandomAction(state); // returns action selection from the default
            roll(state, actions[action]);
            thisDepth++;

        }else if (rolloutType == 1){
            Types.ACTIONS action = actOSLA(state); // returns action from the OSLA action selection Method
            roll(state, action);
            thisDepth++;

        }else if (rolloutType == 2){
            int action = evaluatedRandomAction(state); // returns action from the evaluated random action selection Method
            roll(state, actions[action]);
            thisDepth++;
        }
    }

    return rootStateHeuristic.evaluateState(state);
}
    private int safeRandomAction(GameState state) // The default action selector of MCTS
    {
        Types.TILETYPE[][] board = state.getBoard();
        ArrayList<Types.ACTIONS> actionsToTry = Types.ACTIONS.all();
        int width = board.length;
        int height = board[0].length;

        while(actionsToTry.size() > 0) {

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

        //Uh oh...
        return m_rnd.nextInt(num_actions);
    }

    public Types.ACTIONS actOSLA(GameState state) //Action selection using One Step Look Ahead
    {
// the method below is the implementation of the OSLA for action selection excellent against mcts in 1v1
// but made the agent too slow against 4 players so decided to not use it for main experimentation and tried to create a balance instead.

        ArrayList<Types.ACTIONS> actionsList = Types.ACTIONS.all();
        double maxQ = Double.NEGATIVE_INFINITY;
        Types.ACTIONS bestAction = null;

        for (Types.ACTIONS act : actionsList) {
            GameState gsCopy = state.copy();
            roll(gsCopy, act);
            double valState = rootStateHeuristic.evaluateState(gsCopy);

            double Q = Utils.noise(valState, params.epsilon, this.m_rnd.nextDouble());

            if (Q > maxQ) {
                maxQ = Q;
                bestAction = act;
            }

        }

        return bestAction;

    }



    private int evaluatedRandomAction(GameState state) // This Method was used for the main experimentation.
{

    //List<storedActions> list = new ArrayList(); // This array list would hold the evaluation of the last 5 - 10 actions previously played

    Types.TILETYPE[][] board = state.getBoard();
    ArrayList<Types.ACTIONS> actionsToTry = Types.ACTIONS.all();
    int width = board.length;
    int height = board[0].length;
    double maxQ = Double.NEGATIVE_INFINITY;
    int chosenAction = 0;
    int prevAction = chosenAction;
    while(actionsToTry.size() > 0) {

            int nAction = m_rnd.nextInt(actionsToTry.size());
          //  int rndSafeAction = m_rnd.nextInt(actionsToTry.size());

            Types.ACTIONS act = actionsToTry.get(nAction);
            Types.ACTIONS act2 = actionsToTry.get(prevAction);

            //Types.ACTIONS act3 = actionsToTry.get(rndSafeAction); // code optimisation for speed
        // there is no gain to pick a third random action as it may be worse than the first random action and evaluating and rolling this action will decrease speed of the agent.

            GameState gsCopy = state.copy();
            GameState gsCopy2 = state.copy();

            roll(gsCopy, act);
            roll(gsCopy2, act2);

        double valState = rootStateHeuristic.evaluateState(gsCopy);
        double Q = Utils.noise(valState, params.epsilon, this.m_rnd.nextDouble());
        double prevValState = rootStateHeuristic.evaluateState(gsCopy2);
        double Qprev = Utils.noise(prevValState, params.epsilon, this.m_rnd.nextDouble());


            if (Q > maxQ) {
                maxQ = Q;
                chosenAction = nAction;
                //list.add(new storedActions(nAction,Q)); // tried to store action and evaluation into a list with a wrapper
                // class and retun the highest value action to be played instead of the previous action.
                //since hashmaps only allow a key and a value. i would need a key, action, evaluation.
        }
         else if (Qprev > maxQ ) {
             maxQ = Qprev;
                chosenAction = prevAction;
               act = act2;

            } else {
                chosenAction = nAction;
            }

            Vector2d dir = act.getDirection().toVec();
            Vector2d pos = state.getPosition();
            int x = pos.x + dir.x;
            int y = pos.y + dir.y;

            if (x >= 0 && x < width && y >= 0 && y < height)
                if (board[y][x] != Types.TILETYPE.FLAMES)
                    return chosenAction;


            actionsToTry.remove(nAction);

    }


    //Uh oh...
    return m_rnd.nextInt(num_actions);
}

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
            result = result * Math.pow(gamma,n.m_depth); //discount factor  applied to the result acquired during
            // simulation - the idea was to see how the agent will play when choosing actions
            // from states closer to the root node.
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
