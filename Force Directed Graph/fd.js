// https://github.com/dmesquita/reusable_bubble_chart

var margin  = {top: 10, right: 5, bottom: 10, left: 100},
    width   = 1400-margin.left-margin.right,
    height  = 900-margin.top-margin.bottom;   

var svg = d3.select("body").append("svg")
  .attr("width",width)
  .attr("height",height);

var tooltip = d3.select('body').append('div') .attr("class","tooltip")
      .style({
        'background' : 'orangered',
        'color':'white',
        'width':"90px",
        });

d3.json("test.json",function(data){ 

  var nodes = data["nodes"];
  var links = data["links"];

  var force = d3.layout.force()
    .size([width,height])
    .nodes(d3.values(nodes))
    .links(links)
    .on("tick",tick)

    .linkDistance(300)

    .start();

  var link = svg.selectAll('.link')
    .data(links)
    .enter().append('line')
    .attr("class","link");
  

  var node = d3.select('#flags').selectAll('img')
    .data(force.nodes())
    .enter().append('img')

    .attr('class', function (d) { return 'flag flag-' + d.code; })

    .on('mouseover', mouseoverHandler)
    .on("mousemove",mouseMoving)
    .on("mouseout", mouseoutHandler);
  

  function tick(e){

     node.style('left', function (d) { return d.x + 'px'; })
         .style('top', function (d) { return d.y + 'px'; })
         .call(force.drag);

    link.attr('x1', function(d){ return  d.source.x})
        .attr('y1',function(d){ return  d.source.y})
        .attr('x2', function(d){ return  d.target.x})
        .attr('y2',function(d){ return   d.target.y})
  }
  
  function mouseoverHandler (d) {
     tooltip.transition().style('opacity', .9)
     tooltip.html('<p>' + d["country"] + '</p>' );
  }

  function mouseoutHandler (d) {
      tooltip.transition().style('opacity', 0);
  }

  function mouseMoving (d) {
      tooltip.style("top", (d3.event.pageY-10)+"px").style("left",(d3.event.pageX+10)+"px").style("color","white");
  }
})
