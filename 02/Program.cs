// Mulit linear regression

using TorchSharp;
using static TorchSharp.torch;
using static System.Linq.Enumerable;

var x1_data = new[] { 1, 2, 3 };
var x2_data = new[] { 2, 4, 6 };
var x3_data = new[] { 4, 4, 4 };

var x1 = tensor(x1_data, new[] { (long)x1_data.Length, 1 });
var x2 = tensor(x2_data, new[] { (long)x2_data.Length, 1 });
var x3 = tensor(x3_data, new[] { (long)x3_data.Length, 1 });

var y_data = Range(0, x1_data.Length).Select(i => x1_data[i] + 2 * x2_data[i]).ToArray();
var y = tensor(y_data, new[] { (long)y_data.Length, 1 });

var w1 = zeros(1, requiresGrad: true);
var w2 = zeros(1, requiresGrad: true);
var w3 = zeros(1, requiresGrad: true);
var b = zeros(1, requiresGrad: true);

var optimizer = optim.SGD(new[] { w1, w2, w3, b }, learningRate: 0.01);
var nb_epoches = 2000;
foreach(var epoch in Range(1, nb_epoches))
{
    var hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b;
    var cost = mean(pow(y - hypothesis, 2));
    optimizer.zero_grad();
    cost.backward();
    optimizer.step();

    if (epoch % 100 == 0)
        Console.WriteLine($"Cost: {cost.ToSingle()}");
}
Console.WriteLine($"Hypothesis: {(x1 * w1 + x2 * w2 + x3 * w3 + b).ToString(true)}");
Console.WriteLine($"Result: w1 = {w1.ToSingle()}, w2 = {w2.ToSingle()}, w3 = {w3.ToSingle()}, b = {b.ToSingle()}");
var test_data1 = new[] { 3, 4, 5 };
var test_data2 = new[] { 1, 1, 1 };
var test_data3 = new[] { 9, 8, 534 };
var test1 = tensor(test_data1, new[] { (long)test_data1.Length, 1 });
var test2 = tensor(test_data2, new[] { (long)test_data2.Length, 1 });
var test3 = tensor(test_data3, new[] { (long)test_data3.Length, 1 });

var test_y_data = Range(0, test_data1.Length).Select(i => test_data1[i] + 2 * test_data2[i]).ToArray();
var test_y = tensor(test_y_data, new[] { (long)y_data.Length, 1 });

var h = test1 * w1 + test2 * w2 + test3 * w3;
Console.WriteLine(h.ToString(true));
Console.WriteLine(mean(pow(test_y - h, 2)).ToSingle());