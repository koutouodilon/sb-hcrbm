package sbhcrbm;

import org.openscience.cdk.fingerprint.HybridizationFingerprinter;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.similarity.Tanimoto;
import org.openscience.cdk.smiles.SmilesParser;
import utils.BigFile;
import utils.CSVUtils;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Iterator;

/**
 * Class to calculate the drug-drug chemical similarity
 */
public class ChemicalSimilarity {
    public static void main(String[] args) throws IOException {
        try {
            BigFile f = new BigFile("./datasets/" + "drug_drug_smiles_data.csv");
            String sim_file = "./datasets/drug_drug_smiles_similarity.csv";
            FileWriter writer = new FileWriter(sim_file);
            Iterator<String> iterator = f.iterator();
            /*  We skip the first line */
            if (iterator.hasNext()){
                iterator.next();
                System.out.println("Skip the first line.");
            }
            /* For each drug-drug smiles */
            while (iterator.hasNext()) {
                /* We get and parse the drug-drug smiles */
                String line = iterator.next();
                String[] splits = line.split(",");
                System.out.println(splits[0]);

                String chemical_a_id = splits[0];
                String chemical_a_smiles = splits[1];

                String chemical_b_id = splits[2];
                String chemical_b_smiles = splits[3];

                SmilesParser smilesParser = new SmilesParser(
                        SilentChemObjectBuilder.getInstance()
                );

                IAtomContainer mol1 = smilesParser.parseSmiles(chemical_a_smiles);
                IAtomContainer mol2 = smilesParser.parseSmiles(chemical_b_smiles);
                HybridizationFingerprinter fingerprinter = new HybridizationFingerprinter();
                BitSet bitset1 = fingerprinter.getFingerprint(mol1);
                BitSet bitset2 = fingerprinter.getFingerprint(mol2);
                float tanimotoScore = Tanimoto.calculate(bitset1, bitset2);

                System.out.println("Tanimoto score between drug: " + chemical_a_id + " and drug: " + chemical_b_id + " : " + tanimotoScore);

                ArrayList row = new ArrayList<String>();
                row.add(chemical_a_id);
                row.add(chemical_a_smiles);
                row.add(chemical_b_id);
                row.add(chemical_b_smiles);
                row.add(String.valueOf(tanimotoScore));

                CSVUtils.writeLine(writer, row);
            }
            writer.flush();
            writer.close();
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }
}
