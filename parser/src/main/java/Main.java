import com.github.javaparser.ast.body.*;
import com.github.javaparser.*;

import java.io.FileInputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.*;

public class Main {

    static String delimeter = "====================\n";
    public static void listClasses(String filePath, FileOutputStream os) {
        try {
            new VoidVisitorAdapter<Object>() {
                @Override
                public void visit(MethodDeclaration n, Object arg) {
                    super.visit(n, arg);
                    try {
                        os.write(n.getName().asString().getBytes());
                        os.write("\n".getBytes());
                        os.write(n.getDeclarationAsString().getBytes());
                        os.write(n.getBody().get().toString().getBytes());
                        os.write("\n".getBytes());
                        os.write(delimeter.getBytes());
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
            }.visit(StaticJavaParser.parse(filePath), null);
        } catch (Exception e) {
        }
    }

    private static List<File> getFilesFromDirectory(File directory, int numberOfFiles) {
        File[] filesAndDirectories = directory.listFiles();
        if (filesAndDirectories == null) {
            return new ArrayList<>();
        }
        List<File> files = new ArrayList<>();
        int cnt = 0;
        for (File file : filesAndDirectories) {
            if (file.isDirectory()) {
                files.addAll(getFilesFromDirectory(file, numberOfFiles - cnt));
            } else if (!file.getName().equals(".DS_Store")) {
                files.add(file);
                cnt++;
                if (cnt == numberOfFiles) {
                    return files;
                }
            }
        }
        return files;
    }

    public static void main(String[] args) throws Exception {
        String sourceDirName = args[0];
        int numberOfFilesProcessing = Integer.parseInt(args[1]);
//        String sourceDirName = "../data/java-small/training/spring-framework";
//        Path path = Path.of("../data/java-small/training/spring-framework");
//        System.out.println(path.toAbsolutePath());
//        File tmpfile = new File("/Users/mdmalofeev/Documents/programm/thesis/transformers/data/java-small/training/intellij-community/XmlEnumeratedValueReferenceProvider.java");
//        String text = "";
//        Scanner scanner = new Scanner(tmpfile);
//        while(scanner.hasNextLine()){
//            text+=scanner.nextLine();
//        }
//        int numberOfFilesProcessing = 300;
        //listClasses(text);
        //System.out.println(text);
//        listClasses(text);

        File sourceDir = new File(sourceDirName);
        List<File> files = getFilesFromDirectory(sourceDir, numberOfFilesProcessing);
//        for (File file: files) {
//            System.out.println(file.getAbsolutePath());
//        }
//        listClasses();
////        File file = new File("/Users/mdmalofeev/Documents/programm/thesis/transformers/data/java-small/training/spring-framework/JodaTimeConverters.java");
        try (FileOutputStream fileOutputStream = new FileOutputStream(sourceDirName + "/methods.txt")) {
//            fileOutputStream.write(delimeter.getBytes());
            for (File file : files) {
//                System.out.println(file.getAbsolutePath());
                StringBuilder text = new StringBuilder();
                Scanner scanner = new Scanner(file);
                while (scanner.hasNextLine()) {
                    text.append(scanner.nextLine());
                }
//                System.out.println(text);
                listClasses(text.toString(), fileOutputStream);
            }
        } catch (Exception e) {
            System.err.println("Cannot create the file methods.txt");
        }
    }
}
