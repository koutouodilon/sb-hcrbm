package sbhcrbm;

import com.DTRBM.DTRBMTool;
import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Layers.StochasticBinaryLayer;
import org.jblas.DoubleMatrix;
import utils.CSVUtils;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class DrugTargetCRBMTool extends DTRBMTool {

    private static int _hidUnits = 100;
    private static int _iterNum = 100;
    private static double _learningRate = 0.01;
    private static double _momentum = 0.6;
    private static double _weightCost = 0.00002 * _learningRate;

    /**
     *
     * Method to train (fit) the drug-target CRBM with default values
     *
     * @param data
     * @param prob
     * @param isMissing
     * @param monitoringFolder
     * @return
     */
    public static DrugTargetCRBM fit(double[][][] data, double[][][] prob, boolean[][] isMissing, String monitoringFolder){
        return fit(data, prob, isMissing, _hidUnits, _iterNum, _learningRate, _momentum, _weightCost, monitoringFolder);
    }

    /**
     *
     * Method to train (fit) the drug-target CRBM
     *
     * @param data
     * @param prob
     * @param isMissing
     * @param hidUnits
     * @param iterNum
     * @param learningRate
     * @param momentum
     * @param weightCost
     * @param monitoringFolder
     * @return
     */
    public static DrugTargetCRBM fit(double[][][] data, double[][][] prob, boolean[][] isMissing, int hidUnits, int iterNum, double learningRate, double momentum, double weightCost, String monitoringFolder){
        File dir = new File(monitoringFolder);
        dir.mkdirs();
        String monitoringFile = monitoringFolder + "/monitoring.csv";
        //initialize the RBM model
        int visUnits = data[0][0].length;

        Layer visibleLayer = new StochasticBinaryLayer(visUnits);
        Layer hiddenLayer = new StochasticBinaryLayer(hidUnits);
        visibleLayer.setMomentum(momentum);
        hiddenLayer.setMomentum(momentum);
        visibleLayer.setLearningRate(learningRate);
        hiddenLayer.setLearningRate(learningRate);

        DrugTargetCRBM rbm = new DrugTargetCRBM(visibleLayer, hiddenLayer, learningRate, weightCost, momentum, data.length);

        rbm.setLearningRate(learningRate);
        rbm.setMomentum(momentum);
        double[][] prevConWeightUpdate = new double[rbm.getNumVisibleUnits()][rbm.getNumHiddenUnits()];

        //the length of the binary encoding vector
        int classifyNum = data.length;

        //begin training process

        for(int k = 0; k < iterNum; k++){
            ArrayList<String> arrError = new ArrayList<String>();
            double err = 0;

            for(int i = 0; i < data[0].length; i++){
                double[][] tmp = new double[classifyNum][];
                double[][] tmpProb = new double[classifyNum][];
                for(int j = 0; j < classifyNum; j++){
                    tmp[j] = data[j][i].clone();
                    tmpProb[j] = prob[j][i].clone();
                }
                for(int j = 0; j < isMissing[i].length; j++){
                    if(isMissing[i][j]){
                        for(int m = 0; m < classifyNum; m++){
                            tmp[m][j] = 0.;
                            tmpProb[m][j] = 0.;
                        }
                    }
                }

                err += DrugTargetCDLearner.Learn(rbm, tmp, tmpProb, 1, isMissing[i], prevConWeightUpdate);
            }
            System.out.println("Epoch: " + k + " ; Error: " + err/data[0].length);

            //Training error monitoring
            FileWriter monitoringWriter = null;
            try {
                arrError.add(String.valueOf(k + "," + err/data[0].length));
                monitoringWriter = new FileWriter(monitoringFile,true);
                CSVUtils.writeLine(monitoringWriter, arrError);
                monitoringWriter.flush();
                monitoringWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

        //the successfully trained RBM model
        return rbm;
    }

    /**
     *
     * Prediction algorithm of the RBM model
     *
     * @param crbm
     * @param input
     * @param prob
     * @param isMissing
     * @return
     */
    public static double[][] predict(DrugTargetCRBM crbm, double[][] input, double[][] prob, boolean[] isMissing){

        //input do not satisfy requirement
        if(input.length != crbm.getClassifyNum() || input[0].length != crbm.getVisibleLayer().getNumUnits())
            return null;

        double[][] tmp = new double[input.length][];
        double[][] tmpProb = new double[input.length][];
        for(int i = 0; i < input.length; i++){
            tmp[i] = input[i].clone();
            tmpProb[i] = prob[i].clone();
        }

        for(int i = 0; i < isMissing.length; i++)
            if(isMissing[i]){
                for(int j = 0; j < input.length; j++){
                    tmp[j][i] = 0.;
                    tmpProb[j][i] = 0.;
                }
            }
        crbm.iniInput = tmpProb.clone();
        double[][] hiddenActivity = crbm.getHiddenActivitiesFromVisibleData(tmp);
        double[][] visibleActivity = crbm.getVisibleActivitiesFromHiddenData(hiddenActivity);

        //predicted probability of interactions
        return visibleActivity;
    }

    /**
     *
     * @param folder
     * @param _visUnits
     * @param _hidUnits
     * @param learningRate
     * @param weightCost
     * @param momentum
     * @param _numClass
     * @return
     */
    public static DrugTargetCRBM load(String folder, int _visUnits, int _hidUnits, double learningRate, double weightCost, double momentum, int _numClass){
        int visUnits = _visUnits;
        int hidUnits = _hidUnits;
        int numClass = _numClass;
        Layer visibleLayer = new StochasticBinaryLayer(visUnits);
        Layer hiddenLayer = new StochasticBinaryLayer(hidUnits);

        visibleLayer.setMomentum(momentum);
        hiddenLayer.setMomentum(momentum);
        visibleLayer.setLearningRate(learningRate);
        hiddenLayer.setLearningRate(learningRate);

        DrugTargetCRBM rbm = new DrugTargetCRBM(visibleLayer, hiddenLayer, learningRate, weightCost, momentum, _numClass);
        rbm.setLearningRate(learningRate);
        rbm.setMomentum(momentum);

        try {
            for(int i = 0; i < numClass; i++){
                double[][] currentConnection = DoubleMatrix.loadCSVFile(folder + i + ".W.csv").toArray2();
                rbm.setMultiConnectionWeights(i, currentConnection);
            }

            double[][] conWeight = DoubleMatrix.loadCSVFile(folder + "CW.csv").toArray2();
            rbm.setConWeight(conWeight);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rbm;
    }
}
