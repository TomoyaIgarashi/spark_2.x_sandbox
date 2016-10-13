// ETL
val all_logs = sc.textFile("s3://emr-log-tomoya/click_data_sample.csv")
val header = all_logs.first
val logs = all_logs.filter(_ != header)
val xs1 = logs.map(_.split(","))
val xs2 = xs1.map { case Array(t, c, u) => (t.replace("\"", ""), c.toInt, u.replace("\"", "")) }
val cs1 = xs2.repartition(2).cache()
val ds1 = cs1.toDF
val ds2 = ds1.withColumnRenamed("_1", "accessTime").withColumnRenamed("_2", "userID").withColumnRenamed("_3", "campaignID")
ds2.write.format("com.databricks.spark.csv").option("header", "true").save("s3://emr-log-tomoya/click_data_converted.csv")

// Read CSV
val ds1 = spark.sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("s3://emr-log-tomoya/click_data_converted.csv/*")

ds1.registerTempTable("AccessLog")
val rs1 = sql("SELECT * FROM AccessLog Limit 1")
rs1.show(1, false)
val rs2 = sql("SELECT * FROM AccessLog WHERE accessTime >= '2015-04-27 00:00:00' AND accessTime <= '2015-04-27 00:00:30' ORDER BY accessTime")
rs2.show(18, false)
// Daily Activity
val rs3 = sql("SELECT * FROM AccessLog WHERE accessTime >= '2015-04-27' AND accessTime < date_add('2015-04-27', 1)")
rs3.count
// Daily Active User
val rs4 = sql("SELECT DISTINCT userID FROM AccessLog WHERE accessTime >= '2015-04-27' AND accessTime < date_add('2015-04-27', 1)")
rs4.count
// Retention User
val rs5 = sql("SELECT userID FROM (SELECT userID, count(*) as cnt FROM (SELECT userID, to_date(accessTime) FROM AccessLog WHERE to_date(accessTime) BETWEEN '2015-04-27' AND date_add('2015-04-27', 2) GROUP BY userID, to_date(accessTime)) t1 GROUP BY t1.userID) t2 WHERE t2.cnt = 3")
rs5.count

// Access By User
val rs6 = sql("SELECT count(*) as access_by_user FROM AccessLog GROUP BY userID ORDER BY access_by_user DESC")
val xs1 = rs6.map(_.getLong(0)).collect.toSeq
// ArrayにするためにSeqでSeqを入れ子にする。
val df1 = sc.parallelize(Seq(xs1)).toDF
df1.write.json("s3://emr-log-tomoya/access_by_user.json")
// 追記モード
// df1.write.mode("append").json("s3://emr-log-tomoya/access_by_user.json")
// Output single file
// df1.coalesce(1).write.json("s3://emr-log-tomoya/access_by_user.json")

// Group by access_by_user with count
val gs1 = rs6.groupBy("access_by_user").count
gs1.cache
val xs1 = gs1.select('access_by_user)
val xs2 = gs1.select('count)
val as1 = xs1.map(_.getLong(0)).collect
val as2 = xs2.map(_.getLong(0)).collect
val df1 = sc.parallelize(Array(as1)).toDF
val df2 = sc.parallelize(Array(as2)).toDF
df1.write.json("s3://emr-log-tomoya/access_by_user_label.json")
df2.write.json("s3://emr-log-tomoya/access_by_user_sum.json")

// KMeans access_num_by_user
val rs6 = sql("SELECT count(*) as access_by_user1, count(*) as access_by_user2 FROM AccessLog GROUP BY userID ORDER BY access_by_user1 DESC")
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
val assembler = new VectorAssembler().setInputCols(Array("access_by_user1","access_by_user2")).setOutputCol("features")
val kmeans = new KMeans().setK(2).setFeaturesCol("features").setPredictionCol("prediction")
val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
val kMeansPredictionModel = pipeline.fit(rs6)
val predictionResult = kMeansPredictionModel.transform(rs6)
predictionResult.show

val assembler = new VectorAssembler().setInputCols(Array("access_by_user1","access_by_user2")).setOutputCol("features")
val pipeline = new Pipeline().setStages(Array(assembler))
val pm1 = pipeline.fit(rs6)
val tf1 = pm1.transform(rs6)

val kmeans2 = new KMeans().setK(2).setFeaturesCol("features").setPredictionCol("prediction")
val kmeans3 = new KMeans().setK(3).setFeaturesCol("features").setPredictionCol("prediction")
val kmeans4 = new KMeans().setK(4).setFeaturesCol("features").setPredictionCol("prediction")
val kmeans5 = new KMeans().setK(5).setFeaturesCol("features").setPredictionCol("prediction")
val kmeans6 = new KMeans().setK(6).setFeaturesCol("features").setPredictionCol("prediction")

val km2 = kmeans2.fit(tf1)
val km3 = kmeans3.fit(tf1)
val km4 = kmeans4.fit(tf1)
val km5 = kmeans5.fit(tf1)
val km6 = kmeans6.fit(tf1)

km2.clusterCenters

val WSSSE2 = km2.computeCost(tf1)
val WSSSE3 = km3.computeCost(tf1)
val WSSSE4 = km4.computeCost(tf1)
val WSSSE5 = km5.computeCost(tf1)
val WSSSE6 = km6.computeCost(tf1)

val pr1 = km4.transform(tf1)
pr1.registerTempTable("Clustering4")
val rs9 = sql("SELECT prediction, count(*) as count FROM Clustering4 GROUP BY prediction")

// Access Count By Heavy User
val heavyUserNum4 = rs9.filter($"prediction" > 0).agg(sum($"count")).collect.map(r => r.getLong(0)).head
val accessByHeavyUser4 = sql(s"SELECT userID, count(*) as access_by_user FROM AccessLog GROUP BY userID ORDER BY access_by_user DESC LIMIT $heavyUserNum4")
val gs1 = accessByHeavyUser4.groupBy("access_by_user").count
val xs1 = gs1.select('access_by_user)
val xs2 = gs1.select('count)
val as1 = xs1.map(_.getLong(0)).collect
val as2 = xs2.map(_.getLong(0)).collect
val df1 = sc.parallelize(Array(as1)).toDF
val df2 = sc.parallelize(Array(as2)).toDF
df1.write.json("s3://emr-log-tomoya/access_by_heavy_user_label.json")
df2.write.json("s3://emr-log-tomoya/access_by_heavy_user_sum.json")

// Fetch Heavy User Logs
accessByHeavyUser4.registerTempTable("AccessByHeavyUser")
val heavyUsers = sql("SELECT a.userID as userID, a.campaignID as campaignID FROM AccessLog as a JOIN AccessByHeavyUser as b ON a.userID = b.userID")
