import java.io.IOException;
import java.util.List;
import java.util.StringTokenizer;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class InvertedIndexer {
    public static int min_frequency;  //参数，控制大于某频次的词语
    public static int max_frequency;  //参数，控制小于某频次的词语
    /** 自定义FileInputFormat **/
    public static class FileNameInputFormat extends FileInputFormat<Text, Text> {
        @Override
        public RecordReader<Text, Text> createRecordReader(InputSplit split,
                                                           TaskAttemptContext context) throws IOException, InterruptedException {
            FileNameRecordReader fnrr = new FileNameRecordReader();
            fnrr.initialize(split, context);
            return fnrr;
        }
    }

    /** 自定义RecordReader **/
    public static class FileNameRecordReader extends RecordReader<Text, Text> {
        String fileName;
        LineRecordReader lrr = new LineRecordReader();

        @Override
        public Text getCurrentKey() throws IOException, InterruptedException {
            return new Text(fileName);
        }

        @Override
        public Text getCurrentValue() throws IOException, InterruptedException {
            return lrr.getCurrentValue();
        }

        @Override
        public void initialize(InputSplit arg0, TaskAttemptContext arg1)
                throws IOException, InterruptedException {
            lrr.initialize(arg0, arg1);
            fileName = ((FileSplit) arg0).getPath().getName();
        }

        public void close() throws IOException {
            lrr.close();
        }

        public boolean nextKeyValue() throws IOException, InterruptedException {
            return lrr.nextKeyValue();
        }

        public float getProgress() throws IOException, InterruptedException {
            return lrr.getProgress();
        }
    }


    public static class InvertedIndexMapper extends
            Mapper<Text, Text, Text, IntWritable> {
        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
        }

        protected void map(Text key, Text value, Context context)
                throws IOException, InterruptedException {
            // map()函数这里使用自定义的FileNameRecordReader
            String line = value.toString().toLowerCase();
            StringTokenizer itr = new StringTokenizer(line,"\n");
            for (; itr.hasMoreTokens();) {
                String[] itr1 = itr.nextToken().split(" ");
                int l = itr1.length;
                for(int i = 0;i<l-1;i++){
                    Text word = new Text();
                    word.set(itr1[i]+"#" +itr1[l-1]+" #"+itr1[l-2]);   //词组#url#日期
                    context.write(word, new IntWritable(1));
                }
            }
        }
    }

    /** 使用Combiner将Mapper的输出结果中value部分的词频进行统计 **/
    public static class SumCombiner extends
            Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values)
                sum += val.get();
            result.set(sum);
            context.write(key, result);
        }
    }

    /** 自定义HashPartitioner，保证 <term, docid>格式的key值按照term分发给Reducer **/
    public static class NewPartitioner extends HashPartitioner<Text, IntWritable> {
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {
            String term = key.toString().split("#")[0]; // <term#docid> => term
            return super.getPartition(new Text(term), value, numReduceTasks);
        }
    }

    public static class InvertedIndexReducer extends
            Reducer<Text, IntWritable, Text, Text> {
        private Text word1 = new Text();
        private Text word2 = new Text();
        String temp = new String();
        String date = new String();  ////////////////
        static Text CurrentItem = new Text(" ");
        static List<String> postingList = new ArrayList<String>();

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            //key：词组#日期 #url
            int sum = 0;
            if(key.toString().split("#").length == 3){
                word1.set(key.toString().split("#")[0]);  //词组
                temp = key.toString().split("#")[2];  //url
                date = key.toString().split("#")[1];  //date
                for (IntWritable val : values)
                    sum += val.get();
                word2.set("\n"+date+" "+"<" + temp + "," + sum + ">");
                if (!CurrentItem.equals(word1) && !CurrentItem.equals(" ")) {
                    StringBuilder out = new StringBuilder();
                    long count = 0;
                    for (String p : postingList) {
                        out.append(p);
                        count =
                                count
                                        + Long.parseLong(p.substring(p.indexOf(",") + 1, p.indexOf(">")));
                    }
                    out.append("\n"+"<total:" + count + ">"+"\n");
                    if (count >= min_frequency && count<= max_frequency){
                        context.write(CurrentItem, new Text(out.toString()));
                    }
                    postingList = new ArrayList<String>();
                }
                CurrentItem = new Text(word1);
                postingList.add(word2.toString()); // 不断向postingList也就是文档名称中添加词表
            }
            else{
                return;
            }

        }

        // cleanup 一般情况默认为空，有了cleanup不会遗漏最后一个单词的情况

        public void cleanup(Context context) throws IOException,
                InterruptedException {
            StringBuilder out = new StringBuilder();
            long count = 0;
            for (String p : postingList) {
                out.append(p);
                out.append(";");
                count =
                        count
                                + Long
                                .parseLong(p.substring(p.indexOf(",") + 1, p.indexOf(">")));
            }
            out.append("\n<total:" + count + ">"+"\n");
            if (count >= min_frequency && count<= max_frequency)
                context.write(CurrentItem, new Text(out.toString()));
        }

    }



    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = new Job(conf, "inverted index");
        min_frequency = Integer.parseInt(args[2]);
        max_frequency = Integer.parseInt(args[3]);
        job.setJarByClass(InvertedIndexer.class);
        job.setInputFormatClass(FileNameInputFormat.class);
        job.setMapperClass(InvertedIndexMapper.class);
        job.setCombinerClass(SumCombiner.class);
        job.setReducerClass(InvertedIndexReducer.class);
        job.setPartitionerClass(NewPartitioner.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

