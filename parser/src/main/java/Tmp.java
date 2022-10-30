import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.File;
import java.util.Scanner;

public class Tmp {
    public static void listClasses(String filePath) {


        new VoidVisitorAdapter<Object>() {
            @Override
            public void visit(MethodDeclaration n, Object arg) {
                super.visit(n, arg);
                System.out.println(" * " + n.getBody());
                System.out.println(" * " + n.getDeclarationAsString());
            }
        }.visit(StaticJavaParser.parse(filePath), null);
        System.out.println();
    }

    public static void main(String [] args) throws Exception{


        File file = new File("/Users/mdmalofeev/Documents/programm/thesis/transformers/data/java-small/training/intellij-community/XPathSupportProxy.java");
        String text = "";
        Scanner scanner = new Scanner(file);
        while(scanner.hasNextLine()){
            text+=scanner.nextLine();
        }
        System.out.println(text);
        //listClasses(text);
        //System.out.println(text);
        listClasses(text);

    }
}
