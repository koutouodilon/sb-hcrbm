package test;

import java.io.File;
import java.util.Scanner;

public class TestDrugTargetCRBM {
    public static void main(String[] args){

        Scanner in = new Scanner(System.in);
        int method;

        String[] arguments = new String[17];
        String backup_option;


        System.out.println("Similarity-Boosted Hybrid CRBM");
        System.out.println("MENU");
        System.out.println("Tape 1 to run the T-CRBM(Target-based CRBM) with distinction");
        System.out.println("Tape 2 to run the T-CRBM(Target-based CRBM) without distinction");

        System.out.println("Tape 3 to run the D-CRBM(Drug-based CRBM) with distinction");
        System.out.println("Tape 4 to run the D-CRBM(Drug-based CRBM) without distinction");

        System.out.println("Tape 5 to run the Hybrid-CRBM(combined Drug-based and Target-based CRBM) with distinction");
        System.out.println("Tape 6 to run the Hybrid-CRBM(combined Drug-based and Target-based CRBM) without distinction");


        System.out.println("Tape 7 to run the Similarity-Boosted T-CRBM(SB T-CRBM) with distinction");
        System.out.println("Tape 8 to run the Similarity-Boosted T-CRBM(SB T-CRBM) without distinction");

        System.out.println("Tape 9 to run the Similarity-Boosted D-CRBM(SB D-CRBM) with distinction");
        System.out.println("Tape 10 to run the Similarity-Boosted D-CRBM(SB D-CRBM) without distinction");

        System.out.println("Tape 11 to run the Similarity-Boosted Hybrid-CRBM(SB H-CRBM) with distinction");
        System.out.println("Tape 12 to run the Similarity-Boosted Hybrid-CRBM(SB H-CRBM) without distinction");

        System.out.print("Enter the method you want to run: ");
        method = in.nextInt();

        if(method < 1 || method > 12){
            System.out.println("Invalid method! Exiting...");
            System.exit(-1);
        }

        in = new Scanner(System.in);

        char option;

        if(method != 5 && method != 6 && method != 11 && method != 12){
            System.out.print("Use the trained model if it already exists (y or n): ");
            backup_option = in.nextLine();
            if(backup_option.toLowerCase().charAt(0) != 'y' && backup_option.toLowerCase().charAt(0) != 'n'){
                System.out.println("Invalid trained model option! Exiting...");
                System.exit(-2);
            }
            option = backup_option.toLowerCase().charAt(0);
        }else{
            option = 'y';
        }

        int n_splits = 10;
        String data_folder = "";
        String monitoring_folder = "";
        String backup_folder = "";
        String results_folder = "";
        String drug_rbm_folder = "";
        String target_rbm_folder = "";
        int hidUnits = 100;
        double lr = 0.01;
        double weightCost = 0.00002 * lr;
        int epochs = 100;
        double momentum = 0.6;
        int distinction = 1;
        int numDrugs = 684;
        int numTargets = 1434;
        int numClass = 2;

        switch (method){
            case 1:
                data_folder = "./datasets/crbm/";
                backup_folder = "crbm/distinction/target_rbm/";
                results_folder = "./results/crbm/distinction/target_rbm/";
                monitoring_folder = "./results/crbm/distinction/target_rbm/";
                break;
            case 2:
                data_folder = "./datasets/crbm/";
                backup_folder = "crbm/no_distinction/target_rbm/";
                results_folder = "./results/crbm/no_distinction/target_rbm/";
                monitoring_folder = "./results/crbm/no_distinction/target_rbm/";
                distinction = 2;
                numClass = 1;
                break;
            case 3:
                data_folder = "./datasets/crbm/";
                backup_folder = "crbm/distinction/drug_rbm/";
                results_folder = "./results/crbm/distinction/drug_rbm/";
                monitoring_folder = "./results/crbm/distinction/drug_rbm/";
                break;
            case 4:
                data_folder = "./datasets/crbm/";
                backup_folder = "crbm/no_distinction/drug_rbm/";
                results_folder = "./results/crbm/no_distinction/drug_rbm/";
                monitoring_folder = "./results/crbm/no_distinction/drug_rbm/";
                distinction = 2;
                numClass = 1;
                break;
            case 5:
                data_folder = "./datasets/crbm/";
                drug_rbm_folder = "./backup/crbm/distinction/drug_rbm/";
                target_rbm_folder = "./backup/crbm/distinction/target_rbm/";
                monitoring_folder = "./results/crbm/distinction/hybrid_rbm/";
                results_folder = "./results/crbm/distinction/hybrid_rbm/";
                break;
            case 6:
                data_folder = "./datasets/crbm/";
                drug_rbm_folder = "./backup/crbm/no_distinction/drug_rbm/";
                target_rbm_folder = "./backup/crbm/no_distinction/target_rbm/";
                monitoring_folder = "./results/crbm/no_distinction/hybrid_rbm/";
                results_folder = "./results/crbm/no_distinction/hybrid_rbm/";
                distinction = 2;
                numClass = 2;
                break;
            case 7:
                data_folder = "./datasets/similarity_boosted_omega_0_9/";
                backup_folder = "similarity_boosted_omega_0_9/distinction/target_rbm/";
                results_folder = "./results/similarity_boosted_omega_0_9/distinction/target_rbm/";
                monitoring_folder = "./results/similarity_boosted_omega_0_9/distinction/target_rbm/";
                break;
            case 8:
                data_folder = "./datasets/similarity_boosted_omega_0_9/";
                backup_folder = "similarity_boosted_omega_0_9/no_distinction/target_rbm/";
                results_folder = "./results/similarity_boosted_omega_0_9/no_distinction/target_rbm/";
                monitoring_folder = "./results/similarity_boosted_omega_0_9/no_distinction/target_rbm/";
                distinction = 2;
                numClass = 1;
                break;
            case 9:
                data_folder = "./datasets/similarity_boosted_omega_0_9/";
                backup_folder = "similarity_boosted_omega_0_9/distinction/drug_rbm/";
                results_folder = "./results/similarity_boosted_omega_0_9/distinction/drug_rbm/";
                monitoring_folder = "./results/similarity_boosted_omega_0_9/distinction/drug_rbm/";
                break;
            case 10:
                data_folder = "./datasets/similarity_boosted_omega_0_9/";
                backup_folder = "similarity_boosted_omega_0_9/no_distinction/drug_rbm/";
                results_folder = "./results/similarity_boosted_omega_0_9/no_distinction/drug_rbm/";
                monitoring_folder = "./results/similarity_boosted_omega_0_9/no_distinction/drug_rbm/";
                distinction = 2;
                numClass = 1;
                break;
            case 11:
                data_folder = "./datasets/similarity_boosted_omega_0_9/";
                drug_rbm_folder = "./backup/similarity_boosted_omega_0_9/distinction/drug_rbm/";
                target_rbm_folder = "./backup/similarity_boosted_omega_0_9/distinction/target_rbm/";
                monitoring_folder = "./results/similarity_boosted_omega_0_9/distinction/hybrid_rbm/";
                results_folder = "./results/similarity_boosted_omega_0_9/distinction/hybrid_rbm/";
                break;
            case 12:
                data_folder = "./datasets/similarity_boosted_omega_0_9/";
                drug_rbm_folder = "./backup/similarity_boosted_omega_0_9/no_distinction/drug_rbm/";
                target_rbm_folder = "./backup/similarity_boosted_omega_0_9/no_distinction/target_rbm/";
                monitoring_folder = "./results/similarity_boosted_omega_0_9/no_distinction/hybrid_rbm/";
                results_folder = "./results/similarity_boosted_omega_0_9/no_distinction/hybrid_rbm/";
                distinction = 2;
                numClass = 2;
                break;
        }

        File dir_results_folder = new File(results_folder);
        dir_results_folder.mkdirs();

        File dir_monitoring_folder = new File(monitoring_folder);
        dir_monitoring_folder.mkdirs();

        arguments[0] = Integer.toString(n_splits);
        arguments[1] = data_folder;
        arguments[2] = monitoring_folder;
        arguments[3] = backup_folder;
        arguments[4] = results_folder;
        arguments[5] = Integer.toString(hidUnits);
        arguments[6] = Double.toString(lr);
        arguments[7] = Double.toString(weightCost);
        arguments[8] = Integer.toString(epochs);
        arguments[9] = Double.toString(momentum);
        arguments[10] = Integer.toString(distinction);

        if(method == 5 || method == 6 || method == 11 || method == 12){
            arguments[11] = drug_rbm_folder;
            arguments[12] = target_rbm_folder;
        }
        arguments[13] = Integer.toString(numDrugs);
        arguments[14] = Integer.toString(numTargets);
        arguments[15] = Integer.toString(numClass);
        arguments[16] = String.valueOf(option);

        switch (method) {
            case 1:
            case 2:
                TargetBasedCRBM.main(arguments);
                break;
            case 3:
            case 4:
                DrugBasedCRBM.main(arguments);
                break;
            case 5:
            case 6:
                HybridCRBM.main(arguments);
                break;
            case 7:
            case 8:
                SimilarityBoostedTargetBasedCRBM.main(arguments);
                break;
            case 9:
            case 10:
                SimilarityBoostedDrugBasedCRBM.main(arguments);
                break;
            case 11:
            case 12:
                SimilarityBoostedHybridCRBM.main(arguments);
                break;
        }
    }
}
