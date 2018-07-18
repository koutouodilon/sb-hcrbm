package sbhcrbm;

import com.syvys.jaRBM.Layers.Layer;
import com.syvys.jaRBM.Math.Matrix;
import com.syvys.jaRBM.RBMImpl;
import org.jblas.DoubleMatrix;
import utils.CSVUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


/**
 * CRBM for Drug-Target interaction prediction
 */
public class DrugTargetCRBM extends RBMImpl {
    private static final long serialVersionUID = 1L;

    public boolean[] isMissing;

    //public double[][] conWeight;

    public double[][] iniInput;

    public double[][][] multiVisHidWeights;

    public double[][][] visHidWeightsIncrement;

    public Layer[] visibleLayers;

    public double[][] inputData;

    private int classifyNum;

    public double[][] conWeight;

    public DrugTargetCRBM(Layer VisibleLayer, Layer HiddenLayer, int classifyNum) {
        super(VisibleLayer, HiddenLayer);

        this.classifyNum = classifyNum;
        // TODO Auto-generated constructor stub

        //zero the visible-hidden weights
		/*
		for(int i = 0; i < this.visibleHiddenWeights.length; i++)
			for(int j = 0; j < this.visibleHiddenWeights[i].length; j++)
				this.visibleHiddenWeights[i][j] = 0.;
		*/

        //this.isSemi = true;
        this.multiVisHidWeights = new double[classifyNum][][];
        this.visHidWeightsIncrement = new double[classifyNum][][];
        this.visibleLayers = new Layer[classifyNum];
        for(int i = 0; i < classifyNum; i++){
            this.visibleLayers[i] = this.visibleLayer.clone();
            this.multiVisHidWeights[i] = new double[this.visibleHiddenWeights.length][];
            this.visHidWeightsIncrement[i] = new double[this.visibleHiddenUpdateIncrement.length][];
            for(int j = 0; j < this.visibleHiddenWeights.length; j++){
                this.multiVisHidWeights[i][j] = this.visibleHiddenWeights[j].clone();
                this.visHidWeightsIncrement[i][j] = this.visibleHiddenUpdateIncrement[j].clone();
            }
        }
        conWeight = new double[VisibleLayer.getNumUnits()][HiddenLayer.getNumUnits()];
        conWeight = Matrix.randomizeElements(conWeight);
    }

    public DrugTargetCRBM(Layer VisibleLayer, Layer HiddenLayer, double learningRate,
                  double weightCost, double momentum, int classifyNum) {
        super(VisibleLayer, HiddenLayer, learningRate, weightCost, momentum);
        // TODO Auto-generated constructor stub

        this.classifyNum = classifyNum;

        //zero the visible-hidden weights
		/*
		for(int i = 0; i < this.visibleHiddenWeights.length; i++)
			for(int j = 0; j < this.visibleHiddenWeights[i].length; j++)
				this.visibleHiddenWeights[i][j] = 0.;
		*/

        //this.isSemi = true;
        this.multiVisHidWeights = new double[classifyNum][][];
        this.visHidWeightsIncrement = new double[classifyNum][][];
        this.visibleLayers = new Layer[classifyNum];
        for(int i = 0; i < classifyNum; i++){
            this.visibleLayers[i] = this.visibleLayer.clone();
            this.multiVisHidWeights[i] = new double[this.visibleHiddenWeights.length][];
            this.visHidWeightsIncrement[i] = new double[this.visibleHiddenUpdateIncrement.length][];
            for(int j = 0; j < this.visibleHiddenWeights.length; j++){
                this.multiVisHidWeights[i][j] = this.visibleHiddenWeights[j].clone();
                this.visHidWeightsIncrement[i][j] = this.visibleHiddenUpdateIncrement[j].clone();
            }
        }

        conWeight = new double[VisibleLayer.getNumUnits()][HiddenLayer.getNumUnits()];
        conWeight = Matrix.randomizeElements(conWeight);
    }

    public double[][] getDownwardSWSum(double[][] batchHiddenData, double[][]batchVisibleData){

        return Matrix.multiplyTranspose(batchHiddenData, this.visibleHiddenWeights);

    }

    public double[][] getUpwardSWSum(double[][] batchVisibleData) {

        double[][] tmp = new double[1][this.getHiddenLayer().getNumUnits()];
        //Matrix.zero(tmp);
        int k = 1;
        for(int i = 0; i < batchVisibleData.length; i++){
            double[][] data = {batchVisibleData[i]};
            //tmp = Matrix.add(tmp, Matrix.multiply( data, this.multiVisHidWeights[i]) );
            double[][] ini = {this.iniInput[i]};
//            tmp = Matrix.add(tmp,
//                    Matrix.multiply(data, this.multiVisHidWeights[i]));

            tmp = Matrix.add(tmp,
                    Matrix.add( Matrix.multiply(data, this.multiVisHidWeights[i]),
                            Matrix.multiply(ini, this.conWeight)));
        }

        return tmp;
    }

    public void UpdateVisibleBiases(double[][] data, double[][] negativePhaseVisibleActivities) {

        double[] bias = this.visibleLayer.getBiases().clone();
        double[] biasIncrement = this.visibleLayer.getBiasIncrement().clone();

        this.visibleLayer.updateBiases(data, negativePhaseVisibleActivities);

        double[] biasNew = this.visibleLayer.getBiases().clone();
        double[] biasIncrementNew = this.visibleLayer.getBiasIncrement().clone();

        for(int i= 0; i < this.getNumVisibleUnits(); i++){
            if(this.isMissing[i]) {
                biasNew[i] = bias[i];
                //biasIncrementNew[i] = biasIncrement[i];
                biasIncrementNew[i] = 0;
            }
        }

        this.visibleLayer.setBiases(biasNew);
        this.visibleLayer.setBiasIncrement(biasIncrementNew);
    }

    public double[][] getVisibleActivitiesFromHiddenData(double[][] hiddenData){

        double[][] visActivities = new double[classifyNum][];

        for(int i = 0; i < classifyNum; i++){
            this.setLayerWeight(i);
            visActivities[i] = super.getVisibleActivitiesFromHiddenData(hiddenData)[0].clone();
        }

        return visActivities;
    }

    public void setLayerWeight(int i){
        this.visibleHiddenWeights = this.multiVisHidWeights[i];
        this.visibleLayer = this.visibleLayers[i];
        this.visibleHiddenUpdateIncrement = this.visHidWeightsIncrement[i];
    }

    public void setMultiConnectionWeights(int i, double[][] connectionWeight){
        this.multiVisHidWeights[i] = connectionWeight;
    }

    public void setConWeight(double[][] cWeight){
        this.conWeight = cWeight;
    }

    public void setInitInput(double[][] initInput){
        this.iniInput = initInput;
    }

    public void setHiddenBiases(double[] hbiases){
        this.getHiddenLayer().setBiases(hbiases);
    }

    public void setVisibleBiases(int i, double[] vbiases){
        this.visibleLayers[i].setBiases(vbiases);
    }

    int getClassifyNum() {
        return classifyNum;
    }

    public void save(String folder){
        if (folder == null){
            int timestamp = (int) (new Date().getTime()/1000);
            folder = String.valueOf(timestamp);
        }

        String directory = "./backup/" + folder;
        File dir = new File(directory);
        dir.mkdirs();

        for(int i = 0; i < this.getClassifyNum(); i++){
            DoubleMatrix mWeight = new DoubleMatrix(this.multiVisHidWeights[i]);
            this.saveMatrixToFile(mWeight, directory + "/" + i + ".W.csv");

            DoubleMatrix mVisibleBias = new DoubleMatrix(this.visibleLayers[i].getBiases());
            this.saveMatrixToFile(mVisibleBias, directory + "/" + i + ".VisibleBias.csv");
        }

        DoubleMatrix mHiddenBiases = new DoubleMatrix(this.getHiddenBiases());
        DoubleMatrix mConWeight = new DoubleMatrix(this.conWeight);
        DoubleMatrix mIniInput = new DoubleMatrix(this.iniInput);

        this.saveMatrixToFile(mHiddenBiases, directory + "/HiddenBiases.csv");
        this.saveMatrixToFile(mConWeight, directory + "/CW.csv");
        this.saveMatrixToFile(mIniInput, directory + "/InitInput.csv");

    }

    /**
     * Saves a matrix in a CSV file
     */
    public void saveMatrixToFile(DoubleMatrix matrix, String path){
        try {
            FileWriter writer = new FileWriter(path);

            for(int i = 0; i < matrix.getRows(); i++) {
                List<String> row = new ArrayList<String>();
                for(int j = 0; j < matrix.getColumns(); j++) {
                    String item = String.valueOf(matrix.get(i,j));
                    row.add(item);
                }
                CSVUtils.writeLine(writer, row);
            }

            writer.flush();
            writer.close();

            System.out.println("Writing to matrix finished");

        }catch (IOException e){

        }
    }
}
