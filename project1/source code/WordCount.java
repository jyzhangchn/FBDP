import java.io.IOException;
import java.util.StringTokenizer;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import org.apdplat.word.WordSegmenter;
import org.apdplat.word.segmentation.Word;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.commons.lang3.StringUtils;

public class WordCount {
    public static int min_frequency;  //参数，控制大于某频次的词语
    public static int max_frequency;  //参数，控制小于某频次的词语


    static final String OUT_PATH="side_output/titles.txt"; //新闻标题分词结果以及url写入titles.txt
    static final String OUT_PATH1="input/segment.txt"; //新闻标题分词结果以及url写入titles.txt

    //迭代读取文件夹内所有txt文件
    List<String> pathName = new ArrayList<String>();   //标识文件路径
    public void iteratorPath(String dir) {
        File stock;
        File[] files;
        stock = new File(dir);
        files = stock.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isFile()) {
                    pathName.add(file.getName());
                } else if (file.isDirectory()) {
                    iteratorPath(file.getAbsolutePath());
                }
            }
        }
    }


    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>{
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString()," ,.\":\t\n");
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken().toLowerCase());
                context.write(word, one);
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            if(sum >= min_frequency && sum <= max_frequency ) {  //筛选写出在一定频次之间的词组
                context.write(key, result);
            }
        }
    }

    private static class IntWritableDecreasingComparator extends IntWritable.Comparator {
        //hadoop内置比较函数
        public int compare(WritableComparable a, WritableComparable b) {
            return -super.compare(a, b);
        }
        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            return -super.compare(b1, s1, l1, b2, s2, l2);
        }
    }


    public static void main(String[] args) throws Exception {
        //遍历文件夹将每个txt的新闻标题存入新的文件中

        WordCount news = new WordCount();
        news.iteratorPath("/Users/apple/Documents/FBDP/project1/download_data");
        for (String title : news.pathName) {
            String READ_PATH=("/Users/apple/Documents/FBDP/project1/download_data/"+title);
            //首先读取文件
            ArrayList<String> list = new ArrayList<String>();
            File file = new File(READ_PATH);
            //temp用于需求2url，temp1用于wordcount分词
            BufferedReader br = new BufferedReader(new FileReader(file)); //生成新闻标题+url
            BufferedWriter bw = new BufferedWriter(new FileWriter(OUT_PATH,true)); //输出新闻标题词组+url
            String temp,temp1;
            while((temp=br.readLine())!=null){
                list.add(temp);
            }
            BufferedReader br1 = new BufferedReader(new FileReader(file)); //只生成分词后的新闻标题
            BufferedWriter bw1 = new BufferedWriter(new FileWriter(OUT_PATH1,true)); //输出新闻标题分词
            while((temp1=br.readLine())!=null){
                list.add(temp1);
            }
            br.close();
            br1.close();
            //然后存到数组中
            String[] data; //将新闻文件中的数据以数组的形式存放
            for (int i = 0; i < list.size(); i++) {
                data = list.get(i).split("  ");
                for (int j=3; j<data.length;j+=5 ){
                //对新闻标题分词
                List<Word> tempseg = WordSegmenter.seg(data[j]);
                //把新闻的标题输出到文件
                String titles = StringUtils.strip(tempseg.toString().replace(",",""),"[]");
                bw.write(titles+" "+data[j+1]+" "+data[j-2]);  ///////
                bw1.write(titles);
                bw.newLine();
                bw1.newLine();
                }
            }
            bw.close();
            bw1.close();
        }

        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Path tempDir = new Path("wordcount-temp-output");
        if (otherArgs.length < 4) {
            System.err.println("Usage: wordcount <in> <out> <min_frequency> <max_frequency>");
            System.exit(2);
        }
        min_frequency = Integer.parseInt(otherArgs[otherArgs.length-2]);
        max_frequency = Integer.parseInt(otherArgs[otherArgs.length-1]);
        Job job = new Job(conf, "word count");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        for (int i = 0; i < otherArgs.length - 3; ++i) {
            FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
        }
        FileOutputFormat.setOutputPath(job,tempDir);
        job.waitForCompletion(true);

        Job sortjob = new Job(conf, "sort");
        FileInputFormat.addInputPath(sortjob, tempDir);
        sortjob.setInputFormatClass(SequenceFileInputFormat.class);
        sortjob.setMapperClass(InverseMapper.class);
        sortjob.setNumReduceTasks(1);
        FileOutputFormat.setOutputPath(sortjob,new Path(otherArgs[otherArgs.length - 3]));
        sortjob.setOutputKeyClass(IntWritable.class);
        sortjob.setOutputValueClass(Text.class);
        sortjob.setSortComparatorClass(IntWritableDecreasingComparator.class);
        sortjob.waitForCompletion(true);
        FileSystem.get(conf).delete(tempDir);

        System.exit(0);
    }

}