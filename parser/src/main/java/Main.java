import com.github.javaparser.ast.body.*;
import com.github.javaparser.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.*;

public class Main {

    static String delimeter = "====================\n";

    private static class MethodsVisitor extends VoidVisitorAdapter<Object> {

        MethodsVisitor(FileOutputStream os) {
            this.os = os;
        }
        FileOutputStream os;
        @Override
        public void visit(MethodDeclaration methodDeclaration, Object argument) {
            try {
                super.visit(methodDeclaration, argument);
                os.write(methodDeclaration.getName().asString().getBytes());
                os.write("\n".getBytes());
                os.write(methodDeclaration.getDeclarationAsString().getBytes());
                if (methodDeclaration.getBody().isPresent()) {
                    os.write(methodDeclaration.getBody().get().toString().getBytes());
                }
                os.write("\n".getBytes());
                os.write(delimeter.getBytes());
            } catch (IOException e) {
                System.out.println("Outpit stream went wrong");
            }
        }
    }
    public static void listClasses(String filePath, FileOutputStream os) {
        try {
            MethodsVisitor methodsVisitor = new MethodsVisitor(os);
            methodsVisitor.visit(StaticJavaParser.parse(filePath), null);
        } catch (Exception e){
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
            } else if (!file.getName().equals(".DS_Store")) {  // .DS_Store -- macOS specific files in each directory
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
//         String sourceDirName = "../java-small/test"
//         int numberOfFilesProcessing = 100;
        File sourceDir = new File(sourceDirName);
        List<File> files = getFilesFromDirectory(sourceDir, numberOfFilesProcessing);
        try (FileOutputStream fileOutputStream = new FileOutputStream(sourceDirName + "/methods.txt")) {
            for (File file : files) {
                StringBuilder text = new StringBuilder();
                try (Scanner scanner = new Scanner(file)){
                    while (scanner.hasNextLine()) {
                        text.append(scanner.nextLine());
                    }
                } catch(Exception exception) {
                    System.out.println("Scanner went wrong");
                }
                listClasses(text.toString(), fileOutputStream);
            }
        } catch (Exception e) {
            System.err.println("Cannot create the file methods.txt");
        }
    }
}
