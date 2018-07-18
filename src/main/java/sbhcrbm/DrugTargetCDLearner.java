package sbhcrbm;

import com.DTRBM.DTCDLearner;
import com.syvys.jaRBM.Math.Matrix;
import com.syvys.jaRBM.RBM;

public class DrugTargetCDLearner extends DTCDLearner {

    /**
     * Default constructor
     */
    public DrugTargetCDLearner(){
        super();
    }

    /**
     *
     * Method to learn
     *
     * @param crbm
     * @param data
     * @param prob
     * @param numGibbsIterations
     * @param isMissing
     * @param prevConWeightUpdate
     * @return
     */
    public static double Learn(DrugTargetCRBM crbm, double[][] data, double[][] prob, int numGibbsIterations, boolean[] isMissing, double[][] prevConWeightUpdate) {

        crbm.isMissing = isMissing;
        crbm.iniInput = prob;

        double[][] hiddenActivities = crbm.getHiddenActivitiesFromVisibleData(data);
        // negative phase
        double[][] negPhaseVisible = crbm.getVisibleActivitiesFromHiddenData(hiddenActivities);

        double[][] negPhaseHidden = crbm.getHiddenActivitiesFromVisibleData(negPhaseVisible);

        int num_class = data.length;

        for( int dataType = 0; dataType < num_class; dataType++) {
            //double[][] weightUpdates = getConnectionWeightUpdates(rbm, data, hiddenData, negPhaseVisible, negPhaseHidden);
            //set different visible layers
            crbm.setLayerWeight(dataType);

            double[][] posVis = {data[dataType]}, negVis = {negPhaseVisible[dataType]};

            double[][] weightUpdates = getConnectionWeightUpdates(crbm, posVis, hiddenActivities, negVis, negPhaseHidden);
            for (int i = 0; i < crbm.getNumVisibleUnits(); i++) {
                if (isMissing[i])
                    for (int j = 0; j < crbm.getNumHiddenUnits(); j++)
                        //the weights of the interactions to be predicted is not updated
                        weightUpdates[i][j] = 0.0;
            }
            updateWeights(crbm, weightUpdates);
            crbm.UpdateHiddenBiases(hiddenActivities, negPhaseHidden);
            crbm.UpdateVisibleBiases(posVis, negVis);
        }

        //updating the conditional weights
        double[][] tmp = crbm.conWeight;
        for(int i = 0; i < tmp.length; i++){
            if(isMissing[i])
                continue;
            for(int m = 0; m < data.length; m++){
                if(prob[m][i] > 0.0){
                    for(int k = 0; k < tmp[i].length; k++){
                        prevConWeightUpdate[i][k] = crbm.getMomentum() * prevConWeightUpdate[i][k] + (( hiddenActivities[0][k] - negPhaseHidden[0][k] ) - crbm.getWeightCost() * tmp[i][k]) * crbm.getLearningRate();
                        tmp[i][k] += prevConWeightUpdate[i][k];
                    }
                    break;
                }
            }
        }

        hiddenActivities = crbm.getHiddenActivitiesFromVisibleData(data);
        negPhaseVisible = crbm.getVisibleActivitiesFromHiddenData(hiddenActivities);
        for(int i = 0; i < isMissing.length; i++){
            if(isMissing[i]){
                for(int j = 0; j < negPhaseVisible.length; j++){
                    negPhaseVisible[j][i] = data[j][i];
                }
            }
        }
        return Matrix.getMeanSquaredError(data, negPhaseVisible);
    }

    public static void updateWeights(RBM crbm, double[][] weightUpdates) {
        double[][] weights = crbm.getConnectionWeights();
        for (int v = 0; v < weights.length; v++) {
            for (int h = 0; h < weights[0].length; h++) {
                weights[v][h] += weightUpdates[v][h];
            }
        }
        crbm.setConnectionWeightIncrement(weightUpdates);
    }
}
