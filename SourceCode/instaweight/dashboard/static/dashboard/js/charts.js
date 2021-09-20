var data = {
    series: [5, 3, 4]
}


var sum = function (a, b) {
    return a + b
}

new Chartist.Pie('.pi-chart', data, {
    labelInterpolationFunc: function(value){
        return Math.round(value/data.series.reduce(sum)+100)+'%'
    }
})


new Chartist.Pie('.donut-chart', {
    series: [20, 40, 40]
}, {
        donut: true,
        donutWidth:60,
        donutSolid:true,
        startAngle:270,
        showLabel: true
    }
)


///
var chart = new Chartist.Line('#line-chart', {
    labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    series: [
      [12, 9, 7, 8, 5, 4, 6, 2, 3, 3, 4, 6],
      [4,  5, 3, 7, 3, 5, 5, 3, 4, 4, 5, 5],
      [5,  3, 4, 5, 6, 3, 3, 4, 5, 6, 3, 4],
      
    ]
  }, {
    low: 0
  });
  
  // Let's put a sequence number aside so we can use it in the event callbacks
  var seq = 0,
    delays = 80,
    durations = 500;
  
  
  
    dataDailySalesChart = {
        labels: ['M', 'T', 'W', 'T', 'F', 'S', 'S'],
        series: [
            [12, 17, 7, 17, 23, 18, 38]
        ]
    };

    optionsDailySalesChart = {
        lineSmooth: Chartist.Interpolation.cardinal({
            tension: 0
        }),
        low: 0,
        high: 50, // creative tim: we recommend you to set the high sa the biggest value + something for a better look
        chartPadding: {
            top: 2,
            right: 2,
            bottom: 2,
            left: 2
        },
    }

    var dailySalesChart = new Chartist.Line('#dailySalesChart', dataDailySalesChart, optionsDailySalesChart);




    var data = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        series: [
          [5, 4, 3, 7, 5, 10, 3, 4, 8, 10, 6, 8],
          [3, 2, 9, 5, 4, 6, 4, 6, 7, 8, 7, 4]
        ]
      };
      
      var options = {
        seriesBarDistance: 10
      };
      
      var responsiveOptions = [
        ['screen and (max-width: 640px)', {
          seriesBarDistance: 5,
          axisX: {
            labelInterpolationFnc: function (value) {
              return value[0];
            }
          }
        }]
      ];
      
      new Chartist.Bar('.bar-chart', data, options, responsiveOptions);