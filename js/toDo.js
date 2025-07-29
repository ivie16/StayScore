const todolist =[{
    name:'make',
    due:'01-07-2025'},
    {
    name:'study',
    due:'10-07-2025'}];

renderTodo();
function renderTodo(){
    let todoListHTML ='';
    todolist.forEach((todoObj, index)=> {
        //const todoObj = todolist[i];
        const name= todoObj.name;
        const due = todoObj.due;
        const html = 
        `
            <div>${name}</div> 
            <div>${due}</div> 
            <button onclick="
                todolist.splice(${index},1);
                renderTodo();   
            " class = "delete-button"> Delete </button>
        `;
        todoListHTML+= html;


    }) ;
    /*
    for(let i =0;i< todolist.length;i++)
    {
        const todoObj = todolist[i];
        const name= todoObj.name;
        const due = todoObj.due;
        const html = 
        `
            <div>${name}</div> 
            <div>${due}</div> 
            <button onclick="
                todolist.splice(${i},1);
                renderTodo();   
            " class = "delete-button"> Delete </button>
        `;
        todoListHTML+= html;
    }*/
    document.querySelector('.js-todo-list').innerHTML= todoListHTML;

}


function addTodo(){
    const inputele = document.querySelector('.js-name-input');
    const inputdue= document.querySelector('.js-date-input')
    const name = inputele.value;
    const due = inputdue.value
    todolist.push({name:name, due:due}); 
     
    inputele.value= ''; 
    renderTodo();
}